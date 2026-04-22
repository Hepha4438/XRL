import argparse
import numpy as np
import torch
import gymnasium as gym
import cv2
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Khớp với file train: Resize về 84x84
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

    def observation(self, obs):
        # Render và resize y hệt lúc train
        frame = self.env.render()
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame

def make_vec_env(env_name: str, seed: int = 42):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = PixelObservationWrapper(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    
    # DummyVecEnv nhận vào một list các hàm khởi tạo
    env = DummyVecEnv([_init])
    return VecTransposeImage(env)

def evaluate(model: SAC, env, n_episodes: int, max_steps: int, deterministic: bool):
    successes = 0
    returns = []
    lengths = []

    for i in range(n_episodes):
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

        is_success = (ep_return >= 200.0)
        if is_success:
            successes += 1

        returns.append(ep_return)
        lengths.append(ep_len)
        
        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}/{n_episodes} finished.")

    return {
        "successes": successes,
        "returns": np.array(returns, dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
    }

def main():
    parser = argparse.ArgumentParser(description="Test SAC PixelLunarLanderContinuous model")
    parser.add_argument("--model_path", type=str, default="sac_lunar_lander_continuous.zip")
    parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_false", default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    # Fix lỗi PyTorch 2.6 weights_only
    model = SAC.load(args.model_path, device=args.device, custom_objects={"weights_only": False})

    print(f"Creating env: {args.env_name}")
    env = make_vec_env(args.env_name, seed=args.seed)

    print(f"Evaluating {args.n_episodes} episodes...")
    metrics = evaluate(model, env, args.n_episodes, args.max_steps, args.deterministic)
    env.close()

    successes = metrics["successes"]
    returns = metrics["returns"]
    
    print("\nRESULTS")
    print("=" * 50)
    print(f"Success         : {successes}/{args.n_episodes} ({100.0 * successes / args.n_episodes:.2f}%)")
    print(f"Avg return      : {returns.mean():.4f} +/- {returns.std():.4f}")

if __name__ == "__main__":
    main()