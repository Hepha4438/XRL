#!/usr/bin/env python3
"""
Quick evaluator for PPO Atari Boxing checkpoint.

Usage:
  python test_boxing.py \
      --model_path runs/boxing_s0/final_model \
      --env_name BoxingNoFrameskip-v4 \
      --n_episodes 50
"""

import argparse
import numpy as np
import torch

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage


def make_vec_env(env_name: str, seed: int = 42):
    """
    Tạo môi trường chuẩn cho Atari giống lúc train:
    - make_atari_env (NoFrameskip, wrapper mặc định của SB3 cho Atari)
    - VecFrameStack (gom 4 frames)
    - VecTransposeImage (chuyển channel HWC sang CHW cho PyTorch)
    """
    env = make_atari_env(env_name, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return VecTransposeImage(env)


def load_model(model_path: str, device: str):
    """
    Load PPO model. Không cần custom_objects vì model dùng CnnPolicy mặc định của SB3.
    """
    # Nếu file .zip không có đuôi, tự động thêm để SB3 không bị lỗi
    if not model_path.endswith('.zip') and not os.path.exists(model_path) and os.path.exists(model_path + '.zip'):
        model_path += '.zip'
        
    return PPO.load(model_path, device=device)


def evaluate(model: PPO, env, n_episodes: int, max_steps: int, deterministic: bool):
    successes = 0
    returns = []
    lengths = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done and ep_len < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            ep_len += 1
            done = bool(dones[0])

        # Trong Atari Boxing, không có cờ "is_success".
        # Ta quy ước "Success" là điểm tổng > 0 (bạn đánh trúng nhiều hơn bị đánh)
        is_success = ep_return > 0
        if is_success:
            successes += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    return {
        "successes": successes,
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Test PPO Atari Boxing model")
    # Thay đổi default mặc định về config của Boxing
    parser.add_argument("--model_path", type=str, default="runs/boxing_s0/final_model")
    parser.add_argument("--env_name", type=str, default="BoxingNoFrameskip-v4")
    parser.add_argument("--n_episodes", type=int, default=10)
    # 10000 step là đủ dài cho 1 game Atari
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run evaluation on fixed 5 seeds (42,43,44,45,46) with 1/5 episodes each.")
    return parser.parse_args()


def main():
    import os
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, device=args.device)

    if args.multi_seed:
        # Multi-seed evaluation on 5 seeds
        seeds = [42, 43, 44, 45, 46]
        episodes_per_seed = max(1, args.n_episodes // 5)
        all_results = {}

        print(f"\nMulti-Seed Evaluation: Running on seeds {seeds} with {episodes_per_seed} episodes each")
        print("=" * 70)

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            print(f"\nSeed {seed}...")
            env = make_vec_env(args.env_name, seed=seed)
            metrics = evaluate(
                model=model,
                env=env,
                n_episodes=episodes_per_seed,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
            )
            env.close()

            success_rate = 100.0 * metrics["successes"] / episodes_per_seed
            avg_return = metrics["returns"].mean()
            avg_length = metrics["lengths"].mean()
            all_results[seed] = {"success": success_rate, "return": avg_return, "length": avg_length}

        # Aggregate results
        print(f"\n{'='*70}")
        print("MULTI-SEED AGGREGATED RESULTS")
        print(f"{'='*70}")
        success_rates = [all_results[s]["success"] for s in seeds]
        avg_returns = [all_results[s]["return"] for s in seeds]
        avg_lengths = [all_results[s]["length"] for s in seeds]

        print(f"Win Rate (Score > 0): {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f} %")
        print(f"Avg Return          : {np.mean(avg_returns):.4f} ± {np.std(avg_returns):.4f}")
        print(f"Avg Ep Length       : {np.mean(avg_lengths):.4f} ± {np.std(avg_lengths):.4f}")
        print(f"{'='*70}\n")
    else:
        # Single-seed evaluation
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(f"Creating env: {args.env_name}")
        env = make_vec_env(args.env_name, seed=args.seed)

        print(
            f"Evaluating {args.n_episodes} episodes "
            f"(deterministic={args.deterministic}, max_steps={args.max_steps})"
        )
        metrics = evaluate(
            model=model,
            env=env,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        env.close()

        successes = metrics["successes"]
        returns = metrics["returns"]
        lengths = metrics["lengths"]

        print("\nRESULTS")
        print("=" * 50)
        print(f"Win (Score > 0) : {successes}/{args.n_episodes} ({100.0 * successes / args.n_episodes:.2f}%)")
        print(f"Avg return      : {returns.mean():.4f} +/- {returns.std():.4f}")
        print(f"Avg ep length   : {lengths.mean():.4f} +/- {lengths.std():.4f}")


if __name__ == "__main__":
    main()