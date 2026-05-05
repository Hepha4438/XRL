#!/usr/bin/env python3
"""
Quick evaluator for PPO DoorKey 6x6 checkpoint.

Usage:
  python test_ppo_doorkey_6x6.py \
      --model_path ppo_doorkey_6x6.zip \
      --env_name MiniGrid-DoorKey-6x6-v0 \
      --n_episodes 50
"""

import argparse

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor matching the architecture used during 6x6 training."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float()))


def make_vec_env(env_name: str, seed: int = 42):
    def _make():
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env

        return _init

    env = DummyVecEnv([_make()])
    env.seed(seed)
    return VecTransposeImage(env)


def load_model(model_path: str, device: str):
    # Override policy_kwargs to avoid old cloudpickle incompatibilities.
    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": MinigridFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        }
    }
    return PPO.load(model_path, device=device, custom_objects=custom_objects)


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

        info = infos[0] if infos else {}
        is_success = bool(info.get("is_success", False)) or ep_return > 0
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
    parser = argparse.ArgumentParser(description="Test PPO DoorKey 6x6 model")
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_6x6.zip")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run evaluation on fixed 5 seeds (42,43,44,45,46) with 1/5 episodes each.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, device=args.device)

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

        print(f"Success Rate    : {np.mean(success_rates):.4f} ± {np.std(success_rates):.4f} %")
        print(f"Avg Return      : {np.mean(avg_returns):.4f} ± {np.std(avg_returns):.4f}")
        print(f"Avg Ep Length   : {np.mean(avg_lengths):.4f} ± {np.std(avg_lengths):.4f}")
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
        print(f"Success         : {successes}/{args.n_episodes} ({100.0 * successes / args.n_episodes:.2f}%)")
        print(f"Avg return      : {returns.mean():.4f} +/- {returns.std():.4f}")
        print(f"Avg ep length   : {lengths.mean():.4f} +/- {lengths.std():.4f}")


if __name__ == "__main__":
    main()
