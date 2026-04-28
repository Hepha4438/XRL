import os
import sys
import csv
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup đường dẫn để import được các file cũ
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'experiments', 'lucid'))

from check_success_rules import make_vec_env, ACTION_NAMES
from stable_baselines3 import PPO

from rule_completeness_test import (
    load_model_and_data, 
    get_z_binary, 
    evaluate_pure_boolean_engine, 
    HardRuleAgent,
    CARTPOLE_ACTIONS
)

def calculate_complexity(rules):
    """
    LOGIC SAO CHÉP CHÍNH XÁC 100% TỪ evaluate_lucid_metrics.py
    (Sử dụng feature_pattern.findall để đếm số Literals)
    """
    conjunctive_rule_count = 0
    total_literals = 0
    unique_concepts = set()
    
    # Regex to match concepts like f_12, z_5, etc.
    feature_pattern = re.compile(r'(?:f|z)_\d+')
    
    for action, clauses in rules.items():
        # Remove the bias value string and ignore empty / "(no active clauses)" 
        cleaned_clauses = []
        for clause in clauses:
            c_clean = re.sub(r'\s*\[bias=[^\]]*\]', '', clause).strip()
            if c_clean and c_clean != "(no active clauses)":
                cleaned_clauses.append(c_clean)
                
        # Count each unique clause only once per action
        unique_clauses = set(cleaned_clauses)
        conjunctive_rule_count += len(unique_clauses)
        
        for clause in unique_clauses:
            features = feature_pattern.findall(clause)
            unique_concepts.update(features)
            
            # The number of literals in this clause is the number of features explicitly checked
            total_literals += len(features)
            
    return len(unique_concepts), conjunctive_rule_count, total_literals

def evaluate_online_performance(hard_agent, env_name, episodes, seed):
    """Chạy trực tiếp Agent trong môi trường Gym để lấy Success Rate"""
    env = make_vec_env(env_name, seed=seed)
    successes = 0
    returns = []
    
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not np.any(done):
            pred = hard_agent.predict(obs)
            
            # --- FIX LỖI TUPLE VÀ SHAPE CHO GYMNASIUM ---
            if isinstance(pred, tuple):
                action = pred[0]
            else:
                action = pred
                
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            action = np.array(action).flatten()
            # --------------------------------------------
            
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward[0])
            
        # KHI EPISODE KẾT THÚC (DONE): ĐÁNH GIÁ SUCCESS THEO TỪNG MÔI TRƯỜNG
        if 'CartPole' in env_name:
            if ep_ret >= 500.0:
                successes += 1
        else:
            if ep_ret > 0.0:
                successes += 1
                
        returns.append(ep_ret)
        
    env.close()
    
    success_rate = successes / episodes
    avg_return = np.mean(returns)
    return success_rate, avg_return

def main():
    parser = argparse.ArgumentParser(description="Tau Ablation Study for LUCID")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LUCID model")
    parser.add_argument("--features_path", type=str, required=True, help="Path to offline data")
    parser.add_argument("--env_name", type=str, required=True, help="Gym environment name")
    parser.add_argument("--ppo_path", type=str, required=True, help="Path to Teacher PPO model")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./experiments/tau/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading Models... Device: {device}")
    model, features, actions_tensor, config = load_model_and_data(args.model_path, args.features_path, device)
    
    ppo_model = PPO.load(args.ppo_path, device=device)
    ppo_cnn = ppo_model.policy.features_extractor
    ppo_cnn.eval()

    print("Computing binary concepts (z_binary) for offline Action Fidelity...")
    z_binary = get_z_binary(model, features, device)

    taus = [1/3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    print("\nStarting Tau Sweep...")
    for tau in taus:
        tau_rounded = round(tau, 3)
        print(f"\n[{tau_rounded}] Extracting rules...")
        
        # SỬ DỤNG ACTION_NAMES VÀ GỌI HÀM EXTRACT GIỐNG FILE EVALUATE
        if hasattr(model, 'extract_rules'):
            rules = model.extract_rules(action_names=ACTION_NAMES, threshold=tau)
        else:
            rules = model.logic_layer.extract_rules(action_names=ACTION_NAMES, threshold=tau)
        
        # Tính Complexity (sao chép 100% logic)
        n_concepts, n_rules, n_literals = calculate_complexity(rules)
        
        # Tính Action Fidelity (Offline)
        fallback_idx, af_score = evaluate_pure_boolean_engine(rules, z_binary, actions_tensor, ACTION_NAMES)
        
        # Tính Success Rate (Online)
        hard_agent = HardRuleAgent(ppo_cnn, model, rules, ACTION_NAMES, fallback_action_idx=fallback_idx)
        success_rate, avg_return = evaluate_online_performance(hard_agent, args.env_name, args.episodes, args.seed)
        
        print(f"Tau: {tau_rounded} | Lits: {n_literals} | AF: {af_score*100:.2f}% | SR: {success_rate*100:.2f}% | Ret: {avg_return:.2f}")
        
        results.append({
            "Tau": tau_rounded,
            "Concepts": n_concepts,
            "Rules": n_rules,
            "Literals": n_literals,
            "Action_Fidelity": af_score,
            "Success_Rate": success_rate,
            "Avg_Return": avg_return
        })

    # Lưu CSV
    csv_path = os.path.join(args.output_dir, "tau_ablation.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV to: {csv_path}")

    # Vẽ biểu đồ kết quả (Robustness Flatline Plot)
    x_taus = [r["Tau"] for r in results]
    y_succ = [r["Success_Rate"] * 100 for r in results]
    y_af = [r["Action_Fidelity"] * 100 for r in results]
    
    y_lits = [r["Literals"] for r in results]
    y_rules = [r["Rules"] for r in results]
    y_concs = [r["Concepts"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Binarization Threshold ($\\tau$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.plot(x_taus, y_succ, marker='o', color='tab:blue', linewidth=2, label='Success Rate (Online)')
    ax1.plot(x_taus, y_af, marker='^', color='tab:green', linewidth=2, linestyle='-', label='Action Fidelity (Offline)')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='center left', bbox_to_anchor=(0.0, 0.6))

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Complexity (Count)', fontsize=12, fontweight='bold')  
    
    ax2.plot(x_taus, y_lits, marker='s', color='tab:red', linewidth=2, linestyle='--', label='Total Literals')
    ax2.plot(x_taus, y_rules, marker='d', color='tab:orange', linewidth=2, linestyle=':', label='Total Rules')
    ax2.plot(x_taus, y_concs, marker='X', color='tab:purple', linewidth=2, linestyle='-.', label='Active Concepts')
    
    ax2.tick_params(axis='y')
    
    max_count = max(max(y_lits), max(y_rules), max(y_concs)) if y_lits else 10
    ax2.set_ylim(0, max_count + 5)
    ax2.legend(loc='center right', bbox_to_anchor=(1.0, 0.4))

    plt.title('LUCID Hard Robustness vs. Binarization Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()  
    
    plot_path = os.path.join(args.output_dir, "tau_robustness.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Plot to: {plot_path}")

if __name__ == "__main__":
    main()