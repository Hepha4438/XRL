
"""
evaluate_lucid_metrics.py
Comprehensive evaluation script for LUCID (Soft and Hard) rules.
Combines Complexity metrics, Offline Dataset metrics (Completeness, Fidelity),
and Live Environment metrics across single or multiple seeds.
"""
import sys
import os
import math
import argparse
import re
import json
import numpy as np
import torch
from tqdm import tqdm

from stable_baselines3 import PPO
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# Import môi trường và model architecture
from check_success_rules import make_vec_env
from train_joint import SAELogicAgentV3, SAELogicConfig

ALL_ACTION_NAMES = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
CARTPOLE_ACTIONS = ["Left", "Right"]

# ============================================================================
# CORE LOADING & DATA HANDLING
# ============================================================================
def load_model_and_data(model_path: str, features_path: str, device: str):
    print(f"Loading Logic Model from {model_path}...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = SAELogicConfig(**ckpt['config'])
    model = SAELogicAgentV3(config, device=device)
    model.load_state_dict(ckpt['model_state'])
    model.set_normalization(ckpt['feature_mean'].to(device), ckpt['feature_std'].to(device))
    if 'z_mean' in ckpt:
        model.z_mean.copy_(ckpt['z_mean'])
        model.z_std.copy_(ckpt['z_std'])
    model.to(device)
    model.eval()

    print(f"Loading offline dataset from {features_path}...")
    data = torch.load(features_path, map_location='cpu', weights_only=False)
    features = data['features']
    actions = data['actions']
    return model, features, actions, config

@torch.no_grad()
def get_z_binary(model, features_tensor, device, batch_size=1024):
    model.eval()
    all_z_bin = []
    n = len(features_tensor)
    for i in range(0, n, batch_size):
        batch = features_tensor[i:i+batch_size].to(device)
        batch_norm = model.normalize_input(batch)
        z_sparse, _ = model.sae.encode(batch_norm)
        z_normed = model.normalize_z(z_sparse)
        z_bin = model.bottleneck(z_normed)
        all_z_bin.append(z_bin.cpu())
    return torch.cat(all_z_bin, dim=0).numpy()

# ============================================================================
# COMPLEXITY METRICS
# ============================================================================
def calculate_lucid_complexity(rules):
    conjunctive_rule_count = 0
    total_literals = 0
    unique_concepts = set()
    feature_pattern = re.compile(r'(?:f|z)_\d+')
    
    for action, clauses in rules.items():
        cleaned_clauses = []
        for clause in clauses:
            c_clean = re.sub(r'\s*\[bias=[^\]]*\]', '', clause).strip()
            if c_clean and c_clean != "(no active clauses)":
                cleaned_clauses.append(c_clean)
                
        unique_clauses = set(cleaned_clauses)
        conjunctive_rule_count += len(unique_clauses)
        
        for clause in unique_clauses:
            features = feature_pattern.findall(clause)
            unique_concepts.update(features)
            total_literals += len(features)
            
    return {
        "Conjunctive Rule Count": conjunctive_rule_count,
        "Total Literals": total_literals,
        "Binarized # Concepts": len(unique_concepts)
    }

# ============================================================================
# OFFLINE EVALUATIONS
# ============================================================================
def evaluate_offline_soft(model, features_tensor, actions_tensor, device, batch_size=1024):
    model.eval()
    matches = 0
    n = len(features_tensor)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = features_tensor[i:i+batch_size].to(device)
            batch_norm = model.normalize_input(batch)
            z_sparse, _ = model.sae.encode(batch_norm)
            z_normed = model.normalize_z(z_sparse)
            z_bin = model.bottleneck(z_normed)
            logits = model.logic_layer(z_bin)
            preds = logits.argmax(dim=1).cpu()
            matches += (preds == actions_tensor[i:i+batch_size]).sum().item()
    return (matches / n) * 100.0

def evaluate_offline_hard(rules_dict, z_binary, true_actions, action_names, tau=0.5):
    n_samples = len(z_binary)
    z_bool = z_binary > tau  # Sử dụng trực tiếp tau làm ngưỡng strict boolean
    n_actions = len(action_names)
    
    parsed_rules_list = {act: [] for act in action_names}
    for act_name, clauses in rules_dict.items():
        for clause in clauses:
            if clause == "(no active clauses)": continue
            pos_feats = [int(x) for x in re.findall(r'(?<![¬])f_(\d+)', clause)]
            neg_feats = [int(x) for x in re.findall(r'¬f_(\d+)', clause)]
            bias_match = re.search(r'\[bias=([-\d.]+)\]', clause)
            bias = float(bias_match.group(1)) if bias_match else 0.0
            sig_weight = 1.0 / (1.0 + math.exp(-bias))
            parsed_rules_list[act_name].append((pos_feats, neg_feats, sig_weight))

    action_scores = np.zeros((n_samples, n_actions))
    overall_coverage = np.zeros(n_samples, dtype=bool)

    for act_idx, act_name in enumerate(action_names):
        clauses = parsed_rules_list[act_name]
        for pos_feats, neg_feats, sig_weight in clauses:
            clause_is_true = np.ones(n_samples, dtype=bool)
            for f in pos_feats: clause_is_true &= z_bool[:, f]
            for f in neg_feats: clause_is_true &= ~z_bool[:, f]
            
            action_scores[:, act_idx] += clause_is_true.astype(float) * sig_weight
            overall_coverage |= clause_is_true

    actions_np = true_actions.numpy()
    most_frequent_action_idx = np.bincount(actions_np).argmax()
    predicted_actions = np.full(n_samples, -1, dtype=int)
    
    for i in range(n_samples):
        if overall_coverage[i]:
            predicted_actions[i] = np.argmax(action_scores[i])
        else:
            predicted_actions[i] = most_frequent_action_idx

    completeness_pct = (overall_coverage.sum() / n_samples) * 100
    hard_fidelity_pct = ((predicted_actions == actions_np).sum() / n_samples) * 100

    return most_frequent_action_idx, completeness_pct, hard_fidelity_pct

# ============================================================================
# LIVE GAME AGENTS & EVALUATION
# ============================================================================
class SoftRuleAgent:
    def __init__(self, ppo_cnn, logic_model):
        self.ppo_cnn = ppo_cnn
        self.logic_model = logic_model

    @torch.no_grad()
    def predict(self, obs, device="cpu"):
        device = next(self.ppo_cnn.parameters()).device
        obs_t = torch.as_tensor(obs).float().to(device)
        features = self.ppo_cnn(obs_t)
        features_norm = self.logic_model.normalize_input(features)
        z_sparse, _ = self.logic_model.sae.encode(features_norm)
        z_normed = self.logic_model.normalize_z(z_sparse)
        z_bin = self.logic_model.bottleneck(z_normed)
        logits = self.logic_model.logic_layer(z_bin)
        act = logits.argmax(dim=1).cpu().numpy()
        return act, {"triggered": True}

class HardRuleAgent:
    def __init__(self, ppo_cnn, logic_model, rules_dict, action_names, fallback_action_idx, tau=0.5):
        self.ppo_cnn = ppo_cnn
        self.logic_model = logic_model
        self.action_names = action_names
        self.fallback_action_idx = fallback_action_idx
        self.tau = tau
        self.parsed_rules = self._parse_rules_with_bias(rules_dict)

    def _parse_rules_with_bias(self, rules_dict):
        parsed = {}
        for act_name, clauses in rules_dict.items():
            parsed[act_name] = []
            for clause in clauses:
                if clause == "(no active clauses)": continue
                pos_feats = [int(x) for x in re.findall(r'(?<![¬])f_(\d+)', clause)]
                neg_feats = [int(x) for x in re.findall(r'¬f_(\d+)', clause)]
                bias_match = re.search(r'\[bias=([-\d.]+)\]', clause)
                bias = float(bias_match.group(1)) if bias_match else 0.0
                parsed[act_name].append((pos_feats, neg_feats, bias))
        return parsed

    @torch.no_grad()
    def predict(self, obs, device="cpu"):
        device = next(self.ppo_cnn.parameters()).device
        obs_t = torch.as_tensor(obs).float().to(device)
        
        features = self.ppo_cnn(obs_t)
        features_norm = self.logic_model.normalize_input(features)
        z_sparse, _ = self.logic_model.sae.encode(features_norm)
        z_normed = self.logic_model.normalize_z(z_sparse)
        z_bin = self.logic_model.bottleneck(z_normed)

        z_bool = (z_bin[0] > self.tau).cpu().numpy()
        action_scores = {act: -999.0 for act in self.action_names}
        triggered_any_rule = False

        for act_name, clauses in self.parsed_rules.items():
            act_score = 0.0
            act_triggered = False
            for pos_feats, neg_feats, bias in clauses:
                is_true = True
                for f in pos_feats:
                    if not z_bool[f]: is_true = False; break
                if is_true:
                    for f in neg_feats:
                        if z_bool[f]: is_true = False; break
                
                if is_true:
                    act_score += 1.0 / (1.0 + math.exp(-bias))
                    act_triggered = True
                    triggered_any_rule = True
            
            if act_triggered:
                action_scores[act_name] = act_score

        if not triggered_any_rule:
            return [self.fallback_action_idx], {"triggered": False}

        best_act_name = max(action_scores, key=action_scores.get)
        best_act_idx = self.action_names.index(best_act_name)
        return [best_act_idx], {"triggered": True}

def evaluate_live_game(agent, env_name, n_episodes=100, seed=42):
    env = make_vec_env(env_name, seed=seed)
    successes = 0
    total_rewards = []
    triggered_steps = 0
    total_steps = 0
    
    for ep in tqdm(range(n_episodes), desc=f"Playing {n_episodes} eps (Seed {seed})", leave=False):
        obs = env.reset()
        ep_reward = 0.0
        
        for _ in range(500):
            action, info = agent.predict(obs, device=agent.logic_model.device)
            if info["triggered"]: triggered_steps += 1
            total_steps += 1
            
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
            
            if done[0]:
                if reward[0] > 0: successes += 1
                break
                
        total_rewards.append(ep_reward)
    env.close()
    
    return {
        "Success Rate": (successes / n_episodes) * 100,
        "Average Reward": float(np.mean(total_rewards)),
        "Trigger Rate": (triggered_steps / total_steps) * 100 if total_steps > 0 else 0.0
    }

def run_multi_seed_evaluation(agent, env_name, episodes_per_seed, agent_name="Agent"):
    seeds = [42, 43, 44, 45, 46]
    all_results = {seed: {} for seed in seeds}
    
    print(f"\nEvaluating {agent_name} across 5 seeds (Total {episodes_per_seed * 5} eps)...")
    for seed in seeds:
        metrics = evaluate_live_game(agent, env_name, n_episodes=episodes_per_seed, seed=seed)
        all_results[seed] = metrics
    
    aggregated = {}
    for metric_name in ["Success Rate", "Average Reward", "Trigger Rate"]:
        values = [all_results[s][metric_name] for s in seeds]
        aggregated[metric_name] = {"mean": np.mean(values), "std": np.std(values)}
        
    return aggregated

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="Rule extraction tau")
    parser.add_argument("--env_name", type=str, default="", help="If provided, runs live game eval")
    parser.add_argument("--ppo_path", type=str, default="", help="Path to base PPO model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of total live episodes")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single-seed mode")
    parser.add_argument("--multi-seed", action="store_true", help="Run 5 seeds (1/5 episodes each)")
    parser.add_argument("--save_dir", type=str, default="experiments/lucid/results", help="Directory to save JSON metrics and rules")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. LOAD MODELS AND RULES
    model, features, actions_tensor, config = load_model_and_data(args.model_path, args.features_path, device)
    action_names = CARTPOLE_ACTIONS if config.n_actions == 2 else ALL_ACTION_NAMES[:config.n_actions]
    
    rules = model.logic_layer.extract_rules(action_names=action_names, threshold=args.threshold)
    rules_path = os.path.join(args.save_dir, f"extracted_rules_tau_{args.threshold}.json")
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=4)
    print(f"\n[+] Rules Extracted at tau={args.threshold} (Saved to {rules_path})")

    # 2. COMPLEXITY METRICS
    comp_metrics = calculate_lucid_complexity(rules)
    
    # 3. OFFLINE DATA METRICS
    print(f"\n[+] Computing Offline Metrics on dataset ({len(features)} samples)...")
    soft_af_offline = evaluate_offline_soft(model, features, actions_tensor, device)
    z_binary = get_z_binary(model, features, device)
    fallback_idx, hard_completeness, hard_af_offline = evaluate_offline_hard(
        rules, z_binary, actions_tensor, action_names, tau=args.threshold
    )
    
    # Print Section 1 & 2
    print(f"\n{'='*70}")
    print(f"LUCID COMPREHENSIVE EVALUATION (Tau: {args.threshold})")
    print(f"{'='*70}")
    print("\n--- 1. COMPLEXITY METRICS ---")
    print(f"Conjunctive Rule Count: {comp_metrics['Conjunctive Rule Count']}")
    print(f"Binarized Concepts    : {comp_metrics['Binarized # Concepts']}")
    print(f"Total Literals        : {comp_metrics['Total Literals']}")
    
    print("\n--- 2. OFFLINE DATASET METRICS ---")
    print(f"LUCID SOFT - Action Fidelity (vs PPO): {soft_af_offline:.4f}%")
    print(f"LUCID HARD - Rule Completeness       : {hard_completeness:.4f}%")
    print(f"LUCID HARD - Action Fidelity         : {hard_af_offline:.4f}%")
    print(f"Fallback Action                      : '{action_names[fallback_idx]}'")

    final_metrics = {
        "Tau": args.threshold,
        "Complexity": comp_metrics,
        "Offline": {
            "Soft AF": soft_af_offline,
            "Hard Completeness": hard_completeness,
            "Hard AF": hard_af_offline
        }
    }

    # 4. LIVE GAME METRICS
    if args.env_name and args.ppo_path:
        print("\n--- 3. LIVE ENVIRONMENT METRICS ---")
        ppo_model = PPO.load(args.ppo_path, device=device)
        ppo_cnn = ppo_model.policy.features_extractor
        ppo_cnn.eval()

        soft_agent = SoftRuleAgent(ppo_cnn, model)
        hard_agent = HardRuleAgent(ppo_cnn, model, rules, action_names, fallback_idx, tau=args.threshold)

        if args.multi_seed:
            eps_per_seed = max(1, args.episodes // 5)
            print(f"Mode: MULTI-SEED (5 seeds, {eps_per_seed} episodes/seed)")
            
            soft_live = run_multi_seed_evaluation(soft_agent, args.env_name, eps_per_seed, "LUCID SOFT")
            hard_live = run_multi_seed_evaluation(hard_agent, args.env_name, eps_per_seed, "LUCID HARD")

            print("\n>>> LIVE RESULTS (AGGREGATED MEAN ± STD) <<<")
            for metric in ["Success Rate", "Average Reward", "Trigger Rate"]:
                print(f"SOFT {metric:15s}: {soft_live[metric]['mean']:7.4f} ± {soft_live[metric]['std']:.4f}")
                print(f"HARD {metric:15s}: {hard_live[metric]['mean']:7.4f} ± {hard_live[metric]['std']:.4f}")
                print("-" * 40)
                
            final_metrics["Live"] = {"Mode": "Multi-Seed", "Soft": soft_live, "Hard": hard_live}
        else:
            print(f"Mode: SINGLE-SEED (Seed {args.seed}, {args.episodes} episodes)")
            soft_live = evaluate_live_game(soft_agent, args.env_name, args.episodes, args.seed)
            hard_live = evaluate_live_game(hard_agent, args.env_name, args.episodes, args.seed)

            print("\n>>> LIVE RESULTS (SINGLE SEED) <<<")
            for metric in ["Success Rate", "Average Reward", "Trigger Rate"]:
                print(f"SOFT {metric:15s}: {soft_live[metric]:7.4f}")
                print(f"HARD {metric:15s}: {hard_live[metric]:7.4f}")
                print("-" * 40)
                
            final_metrics["Live"] = {"Mode": "Single-Seed", "Seed": args.seed, "Soft": soft_live, "Hard": hard_live}
    else:
        print("\n--- 3. LIVE ENVIRONMENT METRICS ---")
        print("[!] Skipped. Provide --env_name and --ppo_path to run live evaluations.")

    print(f"\n{'='*70}")
    
    metrics_path = os.path.join(args.save_dir, f"metrics_tau_{args.threshold}.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"[+] All metrics saved to {metrics_path}\n")

if __name__ == "__main__":
    main()