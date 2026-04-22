#!/usr/bin/env python3
"""
Test SAC on Pixel LunarLanderContinuous
"""
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.env.render().shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    def observation(self, obs):
        return self.env.render()

def make_vec_env(env_name: str, seed: int = 42):
    def _make():
        def _init():
            env = gym.make(env_name, render_mode="rgb_array")
            env = PixelObservationWrapper(env)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env
        return _init
    env = DummyVecEnv([_make()])
    env.seed(seed)
    return VecTransposeImage(env)

def evaluate(model: SAC, env, n_episodes: int, max_steps: int, deterministic: bool):
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

        # LunarLander is considered "successful" if reward >= 200
        is_success = (ep_return >= 200.0)
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
    parser = argparse.ArgumentParser(description="Test SAC PixelLunarLanderContinuous model")
    parser.add_argument("--model_path", type=str, default="sac_lunar_lander_continuous.zip")
    parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model_path}")
    model = SAC.load(args.model_path, device=args.device)

    print(f"Creating env: {args.env_name}")
    env = make_vec_env(args.env_name, seed=args.seed)

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