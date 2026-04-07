#!/usr/bin/env python3
"""
Evaluate SAELogicAgentV3 against LUCID baselines by calculating specific complexity
and fidelity metrics as well as typical RL environment performance.
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch
import gymnasium as gym
import minigrid  # noqa: F401
from tqdm import tqdm

# Ensure we can import from the root project directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from check_success_rules import load_rules_agent, make_vec_env, ACTION_NAMES


def calculate_lucid_complexity(agent):
    """
    Compute complexity metrics based on the rules extracted from the logic agent.
    """
    rules = agent.logic_model.extract_rules(action_names=ACTION_NAMES)
    
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
            
    return {
        "Conjunctive Rule Count": conjunctive_rule_count,
        "Total Literals": total_literals,
        "Binarized # Concepts": len(unique_concepts)
    }


def evaluate_lucid_metrics(agent, env_name, n_episodes, max_steps=500, seed=42):
    """
    Run evaluation loop to calculate Score, Average Reward, and Action Fidelity.
    Action Fidelity compares the rule agent's action vs the original PPO's action.
    """
    env = make_vec_env(env_name, seed)
    
    successes = 0
    total_rewards = []
    episode_lengths = []
    
    match_count = 0
    total_steps = 0
    
    pbar = tqdm(total=n_episodes, desc="Evaluating LUCID Metrics")
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        
        for _ in range(max_steps):
            # Rule Agent action
            action = agent.predict(obs)
            
            # PPO Agent action
            ppo_action, _ = agent.ppo_model.predict(obs, deterministic=True)
            
            # Tally matches for Action Fidelity
            if int(action[0]) == int(ppo_action[0]):
                match_count += 1
            total_steps += 1
            
            # Environment step
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            
            if done[0]:
                if reward[0] > 0:
                    successes += 1
                break
                
        total_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        pbar.update(1)
        
    pbar.close()
    env.close()
    
    action_fidelity = (match_count / total_steps) if total_steps > 0 else 0.0
    success_rate = (successes / n_episodes) if n_episodes > 0 else 0.0
    avg_reward = float(np.mean(total_rewards))
    avg_length = float(np.mean(episode_lengths))
    
    return {
        "Game Success Rate": success_rate,
        "Game Avg Return": avg_reward,
        "Game Avg Length": avg_length,
        "Action Fidelity": action_fidelity
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAELogicAgentV3 against LUCID Metrics")
    parser.add_argument("--model_path", type=str,
                        default="./sae_logic_v3_outputs/sae_logic_v3_model.pt",
                        help="Path to trained SAELogicAgentV3 checkpoint")
    parser.add_argument("--ppo_path", type=str,
                        default="ppo_doorkey_6x6.zip",
                        help="Path to PPO base model")
    parser.add_argument("--env_name", type=str,
                        default="MiniGrid-DoorKey-6x6-v0",
                        help="Environment ID")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for environment")
                        
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # 1. Load the agent
    agent = load_rules_agent(args.model_path, args.ppo_path, device)
    
    # 2. Extract Complexity Metrics
    print("\nExtracting Complexity Metrics...")
    complexity_metrics = calculate_lucid_complexity(agent)
    
    # 3. Evaluate Action Fidelity / Environment Performance
    print(f"\nRunning Environment Evaluation on {args.env_name}...")
    env_metrics = evaluate_lucid_metrics(
        agent,
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed
    )
    
    # 4. Print Final Formatted Table
    print("\n--- Metrics ---")
    print(f"Conjunctive Rule Count: {complexity_metrics['Conjunctive Rule Count']}")
    print(f"Binarized Concepts: {complexity_metrics['Binarized # Concepts']}")
    print(f"Total Literals: {complexity_metrics['Total Literals']}")
    print(f"Action Fidelity: {env_metrics['Action Fidelity']:.4f}")
    print(f"Game Success Rate: {env_metrics['Game Success Rate']:.4f}")
    print(f"Game Avg Return: {env_metrics['Game Avg Return']:.4f}")
    print(f"Game Avg Length: {env_metrics['Game Avg Length']:.4f}")
    print("---------------\n")
    
    # 5. Save metrics to JSON
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.json")
    
    final_metrics = {
        "Conjunctive Rule Count": complexity_metrics['Conjunctive Rule Count'],
        "Binarized Concepts": complexity_metrics['Binarized # Concepts'],
        "Total Literals": complexity_metrics['Total Literals'],
        "Action Fidelity": env_metrics['Action Fidelity'],
        "Game Success Rate": env_metrics['Game Success Rate'],
        "Game Avg Return": env_metrics['Game Avg Return'],
        "Game Avg Length": env_metrics['Game Avg Length']
    }
    
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
