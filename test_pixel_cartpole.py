#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from utils_env import make_env_by_name

def make_vec_env(env_name: str, render_mode=None, seed: int = 42):
    def _make():
        def _init():
            env = make_env_by_name(env_name, render_mode=render_mode)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env
        return _init
    env = DummyVecEnv([_make()])
    env.seed(seed)
    return VecTransposeImage(env)

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

        # CartPole is considered "successful" if it survives until max_steps
        is_success = (ep_len >= max_steps)
        if is_success:
            successes += 1

        returns.append(ep_return)
        lengths.append(ep_len)

    return {
        "successes": successes,
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
    }

def main():
    parser = argparse.ArgumentParser(description="Test PPO PixelCartPole model")
    parser.add_argument("--model_path", type=str, default="ppo_cartpole_converted.zip")
    parser.add_argument("--env_name", type=str, default="PixelCartPole-v0")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run evaluation on fixed 5 seeds (42,43,44,45,46) with 1/5 episodes each.")
    args = parser.parse_args()

    # Set basic seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path, device=args.device)

    if args.multi_seed:
        # Multi-seed evaluation on 5 seeds
        seeds = [42, 43, 44, 45, 46]
        episodes_per_seed = args.n_episodes // 5
        all_results = {}

        print(f"\nMulti-Seed Evaluation: Running on seeds {seeds} with {episodes_per_seed} episodes each")
        print("=" * 70)

        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)

            print(f"Seed {seed}...")
            env = make_vec_env(args.env_name, render_mode=None, seed=seed)
            metrics = evaluate(model, env, episodes_per_seed, args.max_steps, args.deterministic)
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

        print(f"Success Rate    : {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f} %")
        print(f"Avg Return      : {np.mean(avg_returns):.4f} ± {np.std(avg_returns):.4f}")
        print(f"Avg Ep Length   : {np.mean(avg_lengths):.4f} ± {np.std(avg_lengths):.4f}")
        print(f"{'='*70}\n")
    else:
        # Single-seed evaluation
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(f"Creating env: {args.env_name}")
        env = make_vec_env(args.env_name, render_mode=None, seed=args.seed)

        print(f"Evaluating {args.n_episodes} episodes (deterministic={args.deterministic})")
        metrics = evaluate(model, env, args.n_episodes, args.max_steps, args.deterministic)
        env.close()

        successes = metrics["successes"]
        returns = metrics["returns"]
        lengths = metrics["lengths"]

        print("\nRESULTS")
        print("=" * 50)
        print(f"Success         : {successes}/{args.n_episodes} ({100.0 * successes / args.n_episodes:.2f}%)")
        print(f"Avg return      : {returns.mean():.4f} +/- {returns.std():.4f}")
        print(f"Avg ep length   : {lengths.mean():.1f} +/- {lengths.std():.1f}")

if __name__ == "__main__":
    main()
