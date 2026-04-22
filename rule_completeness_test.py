#!/usr/bin/env python3
"""
rule_completeness_test.py
Evaluates the Completeness and STRICT BOOLEAN Fidelity of extracted DNF rules.
NEW: Includes a "Hard Rule Agent" mode that plays the live Gym environment 
using Strict Boolean Logic + Bias Scoring.
"""
import math
import argparse
import re
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO

# Sửa import: Dùng make_vec_env chính chủ của bạn để hỗ trợ custom env (PixelCartPole)
from check_success_rules import make_vec_env
from train_sae_logic import SAELogicAgentV3, SAELogicConfig

ALL_ACTION_NAMES = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
CARTPOLE_ACTIONS = ["Left", "Right"]

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

    print(f"Loading dataset from {features_path}...")
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
# HARD RULE AGENT (FOR LIVE GAME PLAY)
# ============================================================================
class HardRuleAgent:
    def __init__(self, ppo_cnn, logic_model, rules_dict, action_names, fallback_action_idx):
        self.ppo_cnn = ppo_cnn
        self.logic_model = logic_model
        self.action_names = action_names
        self.fallback_action_idx = fallback_action_idx
        self.parsed_rules = self._parse_rules_with_bias(rules_dict)

    def _parse_rules_with_bias(self, rules_dict):
        """Dịch luật Text thành cấu trúc: (pos_feats, neg_feats, bias)"""
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
        obs_t = torch.as_tensor(obs).float().to(device)
        
        # 1. Đi qua PPO CNN và SAE để lấy Z_binary
        features = self.ppo_cnn(obs_t)
        features_norm = self.logic_model.normalize_input(features)
        z_sparse, _ = self.logic_model.sae.encode(features_norm)
        z_normed = self.logic_model.normalize_z(z_sparse)
        z_bin = self.logic_model.bottleneck(z_normed)

        # 2. Strict Boolean Threshold
        z_bool = (z_bin[0] > 0.5).cpu().numpy()

        # 3. Tính điểm (Score) cho từng Action dựa trên Bias của các luật đạt True
        action_scores = {act: -999.0 for act in self.action_names} # Khởi tạo điểm rất thấp
        triggered_any_rule = False

        for act_name, clauses in self.parsed_rules.items():
            act_score = 0.0
            act_triggered = False
            for pos_feats, neg_feats, bias in clauses:
                # Kiểm tra luật theo Hard Logic
                is_true = True
                for f in pos_feats:
                    if not z_bool[f]: is_true = False; break
                if is_true:
                    for f in neg_feats:
                        if z_bool[f]: is_true = False; break
                
                # Nếu luật thỏa mãn, cộng Bias vào điểm của Action đó
                if is_true:
                    sigmoid_weight = 1.0 / (1.0 + math.exp(-bias))
                    act_score += sigmoid_weight
                    act_triggered = True
                    triggered_any_rule = True
            
            if act_triggered:
                action_scores[act_name] = act_score

        # 4. Trả về Action có Score cao nhất, hoặc Fallback nếu không có luật nào sáng
        if not triggered_any_rule:
            return [self.fallback_action_idx], {"triggered": False}

        best_act_name = max(action_scores, key=action_scores.get)
        best_act_idx = self.action_names.index(best_act_name)
        return [best_act_idx], {"triggered": True}

def evaluate_live_game(agent, env_name, n_episodes=100, seed=42):
    # Đã xóa n_envs=1 để tương thích với hàm make_vec_env của bạn
    env = make_vec_env(env_name, seed=seed)
    successes = 0
    total_rewards = []
    
    triggered_steps = 0
    total_steps = 0
    
    print(f"\n[Live Game] Playing {n_episodes} episodes using HARD RULES + BIAS SCORING (Seed: {seed})...")
    for ep in tqdm(range(n_episodes)):
        obs = env.reset()
        ep_reward = 0.0
        
        for _ in range(500): # Max steps
            action, info = agent.predict(obs, device=agent.logic_model.device)
            if info["triggered"]:
                triggered_steps += 1
            total_steps += 1
            
            obs, reward, done, _ = env.step(action)
            ep_reward += reward[0]
            
            if done[0]:
                if reward[0] > 0: successes += 1
                break
                
        total_rewards.append(ep_reward)
        
    env.close()
    
    success_rate = (successes / n_episodes) * 100
    avg_reward = np.mean(total_rewards)
    rule_trigger_rate = (triggered_steps / total_steps) * 100 if total_steps > 0 else 0.0
    
    print(f"\n{'='*70}")
    print("LIVE GAME PERFORMANCE (HARD RULE + BIAS AGENT)")
    print(f"{'='*70}")
    print(f"Success Rate       : {success_rate:.2f}%")
    print(f"Average Reward     : {avg_reward:.4f}")
    print(f"Live Trigger Rate  : {rule_trigger_rate:.2f}% (Steps where >= 1 rule fired)")
    print(f"{'='*70}\n")


# ============================================================================
# OFFLINE COMPLETENESS EVALUATION
# ============================================================================
def evaluate_pure_boolean_engine(rules_dict, z_binary, true_actions, action_names):
    n_samples = len(z_binary)
    z_bool = z_binary > 0.5 
    n_actions = len(action_names)
    
    # 1. Tiền xử lý luật: Chuyển text thành chỉ số và bias để tính toán vector hóa
    parsed_rules_list = {act: [] for act in action_names}
    for act_name, clauses in rules_dict.items():
        for clause in clauses:
            if clause == "(no active clauses)": continue
            pos_feats = [int(x) for x in re.findall(r'(?<![¬])f_(\d+)', clause)]
            neg_feats = [int(x) for x in re.findall(r'¬f_(\d+)', clause)]
            bias_match = re.search(r'\[bias=([-\d.]+)\]', clause)
            bias = float(bias_match.group(1)) if bias_match else 0.0
            # Tính trước Sigmoid Bias
            sig_weight = 1.0 / (1.0 + math.exp(-bias))
            parsed_rules_list[act_name].append((pos_feats, neg_feats, sig_weight))

    # 2. Tính toán Score cho từng Action trên toàn bộ tập dữ liệu
    # Shape: [n_samples, n_actions]
    action_scores = np.zeros((n_samples, n_actions))
    overall_coverage = np.zeros(n_samples, dtype=bool)

    for act_idx, act_name in enumerate(action_names):
        clauses = parsed_rules_list[act_name]
        for pos_feats, neg_feats, sig_weight in clauses:
            # Kiểm tra xem clause có thỏa mãn logic Boolean không
            clause_is_true = np.ones(n_samples, dtype=bool)
            for f in pos_feats: clause_is_true &= z_bool[:, f]
            for f in neg_feats: clause_is_true &= ~z_bool[:, f]
            
            # Nếu luật True, cộng dồn Sigmoid Weight vào Score của Action đó
            action_scores[:, act_idx] += clause_is_true.astype(float) * sig_weight
            overall_coverage |= clause_is_true

    # 3. Quyết định Action cuối cùng
    actions_np = true_actions.numpy()
    most_frequent_action_idx = np.bincount(actions_np).argmax()
    
    predicted_actions = np.full(n_samples, -1, dtype=int)
    
    for i in range(n_samples):
        if overall_coverage[i]:
            # Chọn Action có tổng điểm Bias cao nhất (giống Live Agent và Theorem 2)
            predicted_actions[i] = np.argmax(action_scores[i])
        else:
            # Nếu không có luật nào phủ, dùng Fallback (Action phổ biến nhất)
            predicted_actions[i] = most_frequent_action_idx

    # 4. Thống kê
    covered_count = overall_coverage.sum()
    completeness_pct = (covered_count / n_samples) * 100
    boolean_fidelity_pct = (predicted_actions == actions_np).sum() / n_samples * 100

    print(f"\n{'='*70}")
    print("OFFLINE: SYMBOLIC ENGINE (Strict Boolean + Bias Weighting)")
    print(f"{'='*70}")
    print(f"Total States Evaluated : {n_samples}")
    print(f"OVERALL COMPLETENESS   : {completeness_pct:.2f}%")
    print(f"STRICT BOOLEAN FIDELITY: {boolean_fidelity_pct:.2f}%")
    print(f"Fallback Action used   : '{action_names[most_frequent_action_idx]}'")
    print(f"{'='*70}\n")
    
    return most_frequent_action_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="Rule extraction threshold")
    parser.add_argument("--env_name", type=str, default="", help="If provided, runs live game eval")
    parser.add_argument("--ppo_path", type=str, default="", help="Path to base PPO model (required for live eval)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of live episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model, features, actions_tensor, config = load_model_and_data(args.model_path, args.features_path, device)
    
    action_names = CARTPOLE_ACTIONS if config.n_actions == 2 else ALL_ACTION_NAMES[:config.n_actions]
    rules = model.logic_layer.extract_rules(action_names=action_names, threshold=args.threshold)
    
    print(f"\n[+] Extracted Rules with Threshold = {args.threshold}")
    
    # 1. Chạy đánh giá Offline
    print("Computing binary concepts (z_binary) for all offline samples...")
    z_binary = get_z_binary(model, features, device)
    fallback_idx = evaluate_pure_boolean_engine(rules, z_binary, actions_tensor, action_names)

    # 2. Chạy Live Game nếu có đủ tham số
    if args.env_name and args.ppo_path:
        print(f"\nLoading PPO model from {args.ppo_path} for CNN feature extraction...")
        ppo_model = PPO.load(args.ppo_path, device=device)
        ppo_cnn = ppo_model.policy.features_extractor
        ppo_cnn.eval()

        hard_agent = HardRuleAgent(
            ppo_cnn=ppo_cnn, 
            logic_model=model, 
            rules_dict=rules, 
            action_names=action_names, 
            fallback_action_idx=fallback_idx
        )
        
        # Truyền args.seed vào hàm đánh giá
        evaluate_live_game(hard_agent, args.env_name, n_episodes=args.episodes, seed=args.seed)
    else:
        print("\n[!] Skipping Live Game evaluation. Provide --env_name and --ppo_path to enable.")

if __name__ == "__main__":
    main()