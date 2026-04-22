"""
VIPER Baseline Implementation
(Verifiable Reinforcement Learning via Policy Extraction - Bastani et al. 2018)

ACTUAL VIPER IMPLEMENTATION:
1. Interactive DAgger (Dataset Aggregation) Loop.
2. Cost-Sensitive Learning via Original Resampling (Weighted by Teacher's Action Probability Gap).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import torch
import gymnasium as gym
import minigrid
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACTUAL VIPER Baseline (DAgger + Original Resampling)")

    parser.add_argument("--ppo_path", type=str, default="ppo_doorkey_6x6.zip",
                        help="Path to the trained PPO Teacher model.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0",
                        help="Gym environment name.")
    parser.add_argument("--save_dir", type=str, default="experiments/dt/results/",
                        help="Directory to save the best model and metrics.")
    
    # VIPER specific hyperparameters
    parser.add_argument("--n_dagger_iters", type=int, default=15,
                        help="Number of DAgger iterations.")
    parser.add_argument("--episodes_per_iter", type=int, default=20,
                        help="Number of episodes to rollout per DAgger iteration.")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of the Decision Tree.")
    parser.add_argument("--min_samples_leaf", type=int, default=5,
                        help="Minimum samples per leaf to prevent overfitting.")
    parser.add_argument("--n_eval_episodes", type=int, default=1000,
                        help="Number of episodes for the FINAL rigorous evaluation.")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode (determines success in CartPole).")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================================
# AGENT WRAPPERS
# ============================================================================

class DTAgent:
    """Student Agent: Uses a Decision Tree on top of PPO's frozen CNN."""
    def __init__(self, dt_model: DecisionTreeClassifier, ppo_model: PPO, device: str):
        self.dt_model = dt_model
        self.ppo_model = ppo_model
        self.device = device
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).float().to(self.device)
            features = self.ppo_model.policy.features_extractor(obs_tensor)
        
        # Scikit-learn expects 2D array: (batch_size, features_dim)
        action = self.dt_model.predict(features.cpu().numpy())
        return action

class PPOTeacherAgent:
    """Teacher Agent: Original PPO policy."""
    def __init__(self, ppo_model: PPO):
        self.ppo_model = ppo_model
        
    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        return action


# ============================================================================
# CORE VIPER LOGIC (DAgger + Labeling + Weighting)
# ============================================================================

def query_teacher(ppo_model: PPO, obs: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Passes observations through the Teacher to get:
    1. Features (CNN output)
    2. Best Action (Label)
    3. Criticality Weight (Max Prob - Min Prob) for Cost-Sensitive Learning
    """
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs).float().to(device)
        
        # 1. Get CNN Features
        features = ppo_model.policy.features_extractor(obs_tensor)
        
        # 2. Forward through MLP to get Action Logits
        latent_pi, _ = ppo_model.policy.mlp_extractor(features)
        logits = ppo_model.policy.action_net(latent_pi)
        probs = torch.softmax(logits, dim=-1)
        
        # 3. Label and Weight
        best_actions = logits.argmax(dim=-1).cpu().numpy()
        
        # Cost-sensitive weight: Difference between most likely and least likely action
        weights = (probs.max(dim=-1)[0] - probs.min(dim=-1)[0]).cpu().numpy()
        
    return features.cpu().numpy(), best_actions, weights


def collect_rollouts(env, agent, teacher_ppo: PPO, n_episodes: int, device: str, max_steps: int):
    """
    Rollout the given agent in the environment. 
    Regardless of who is playing (Teacher or Student), the data is ALWAYS labeled by the Teacher.
    """
    all_features = []
    all_actions = []
    all_weights = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_len = 0
        
        while not done and ep_len < max_steps:
            # 1. Ask Teacher to analyze the current state
            feats, teacher_acts, weights = query_teacher(teacher_ppo, obs, device)
            
            all_features.append(feats[0])
            all_actions.append(teacher_acts[0])
            all_weights.append(weights[0])
            
            # 2. Agent decides the next step
            action = agent.predict(obs)
            obs, _, dones, _ = env.step(action)
            
            done = bool(dones[0])
            ep_len += 1
            
    return all_features, all_actions, all_weights


def evaluate_agent(env, agent, n_episodes: int, env_name: str, max_steps: int) -> Tuple[float, float, float]:
    """Evaluation of an agent's success rate, return, and length."""
    successes = 0
    returns = []
    lengths = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        
        while not done and ep_len < max_steps:
            action = agent.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            done = bool(dones[0])
            ep_len += 1
            
        # Phân biệt tiêu chí thành công giữa CartPole và MiniGrid
        if "CartPole" in env_name:
            is_success = (ep_len >= max_steps)
        else:
            info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else (infos[0] if infos else {})
            is_success = bool(info.get("is_success", False)) or ep_return > 0
            
        if is_success:
            successes += 1
            
        returns.append(ep_return)
        lengths.append(ep_len)
            
    return successes / n_episodes, float(np.mean(returns)), float(np.mean(lengths))


# ============================================================================
# METRICS & UTILS
# ============================================================================

def calculate_tree_metrics(clf: DecisionTreeClassifier) -> Dict[str, Any]:
    tree = clf.tree_
    rule_set_cardinality = clf.get_n_leaves()
    binarized_concepts = tree.node_count - rule_set_cardinality

    total_literals = 0
    stack = [(0, 0)] 
    while stack:
        node_id, depth = stack.pop()
        if tree.children_left[node_id] != -1:
            stack.append((tree.children_left[node_id], depth + 1))
            stack.append((tree.children_right[node_id], depth + 1))
        else:
            total_literals += depth

    return {
        "Rule Set Cardinality": int(rule_set_cardinality),
        "Binarized Concepts": int(binarized_concepts),
        "Total Literals": int(total_literals),
        "Tree Depth": int(clf.get_depth()),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Starting VIPER with DAgger. Device: {device}")

    # 1. Environment Setup
    def _init():
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from utils_env import make_env_by_name
        return make_env_by_name(args.env_name, seed=args.seed)
        
    env = VecTransposeImage(DummyVecEnv([_init]))
    eval_env = VecTransposeImage(DummyVecEnv([_init])) 
    
    # 2. Load Teacher PPO
    print(f"Loading PPO Teacher from {args.ppo_path}...")
    try:
        ppo_teacher = PPO.load(args.ppo_path, device=device)
    except Exception as e:
        print(f"Fallback loading custom PPO... Error: {e}")
        class MinigridFeaturesExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                c = observation_space.shape[0]
                self.cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(c, 32, 3, 1, 1), torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 3, 1, 1), torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 3, 1, 1), torch.nn.ReLU(),
                    torch.nn.Flatten(),
                )
                with torch.no_grad():
                    n_flat = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
                self.linear = torch.nn.Sequential(torch.nn.Linear(n_flat, features_dim), torch.nn.ReLU())
            def forward(self, obs): return self.linear(self.cnn(obs.float()))
        ppo_teacher = PPO.load(args.ppo_path, device=device, custom_objects={
            "policy_kwargs": {"features_extractor_class": MinigridFeaturesExtractor, "features_extractor_kwargs": {"features_dim": 128}}
        })

    teacher_agent = PPOTeacherAgent(ppo_teacher)
    
    # 3. VIPER / DAgger Loop
    dataset_features = []
    dataset_actions = []
    dataset_weights = []
    
    best_success_rate = -1.0
    best_clf = None
    
    print("\n" + "="*50)
    print("VIPER ITERATION 0: TEACHER DEMONSTRATIONS")
    print("="*50)
    f, a, w = collect_rollouts(env, teacher_agent, ppo_teacher, args.episodes_per_iter * 2, device, args.max_steps)
    dataset_features.extend(f)
    dataset_actions.extend(a)
    dataset_weights.extend(w)
    
    for i in range(1, args.n_dagger_iters + 1):
        print(f"\n" + "="*50)
        print(f"VIPER ITERATION {i}/{args.n_dagger_iters}")
        print("="*50)
        
        # A. Original VIPER Resampling Logic
        X = np.vstack(dataset_features)
        y = np.array(dataset_actions)
        w = np.array(dataset_weights)
        
        # Normalize weights to form a probability distribution
        w_sum = np.sum(w)
        if w_sum > 0:
            p = w / w_sum
        else:
            p = np.ones_like(w) / len(w) # Fallback to uniform distribution
            
        # Sample with replacement
        n_samples = len(y)
        print(f"  -> Resampling {n_samples} transitions based on Teacher's probability gap...")
        
        # Set seed for reproducibility during resampling
        rng = np.random.default_rng(args.seed + i)
        indices = rng.choice(n_samples, size=n_samples, replace=True, p=p)
        
        X_resampled = X[indices]
        y_resampled = y[indices]
        
        print(f"  -> Training DT on resampled dataset...")
        clf = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            ccp_alpha=0.001,
            random_state=args.seed + i,
        )
        # Train normally without sample_weight (already accounted for via resampling)
        clf.fit(X_resampled, y_resampled) 
        
        # B. Evaluate current Student
        student_agent = DTAgent(clf, ppo_teacher, device)
        eval_eps = 20
        print(f"  -> Evaluating Student on {eval_eps} episodes...")
        success_rate, _, _ = evaluate_agent(eval_env, student_agent, eval_eps, args.env_name, args.max_steps)
        print(f"  -> Student Success Rate: {success_rate * 100:.1f}%")
        
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_clf = clf
            print("  -> [*] New Best Model!")
            
        # C. Rollout Student to collect DAgger data
        if i < args.n_dagger_iters:
            print(f"  -> Student exploring the environment to find edge cases...")
            f, a, w = collect_rollouts(env, student_agent, ppo_teacher, args.episodes_per_iter, device, args.max_steps)
            dataset_features.extend(f)
            dataset_actions.extend(a)
            dataset_weights.extend(w)
            print(f"  -> Added {len(a)} new samples to dataset.")

    # 4. Final Evaluation & Save
    print("\n" + "="*50)
    print("FINAL VIPER EVALUATION")
    print("="*50)
    
    print(f"Running robust final evaluation on {args.n_eval_episodes} episodes...")
    final_student = DTAgent(best_clf, ppo_teacher, device)
    final_success_rate, final_avg_return, final_avg_length = evaluate_agent(eval_env, final_student, args.n_eval_episodes, args.env_name, args.max_steps)
    
    # Calculate Action Fidelity on the full aggregated (un-resampled) dataset
    X_full = np.vstack(dataset_features)
    y_full = np.array(dataset_actions)
    y_pred = best_clf.predict(X_full)
    action_fidelity = accuracy_score(y_full, y_pred)
    
    metrics = calculate_tree_metrics(best_clf)
    metrics["Action Fidelity"] = float(action_fidelity)
    metrics["Game Success Rate"] = float(final_success_rate)
    metrics["Game Avg Return"] = float(final_avg_return)
    metrics["Game Avg Length"] = float(final_avg_length)

    print("\n--- Metrics ---")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")
    print("---------------\n")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "viper_policy.pkl"
    joblib.dump(best_clf, model_path)
    
    metrics_path = save_dir / "viper_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved best VIPER model to {model_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()