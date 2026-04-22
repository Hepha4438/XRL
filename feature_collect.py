"""
Data Collection Script for Continuous Actions (SAC/PPO)
No SVD/ICA analysis included. Just outputs raw data for Stage 2.
"""
import argparse
import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.env.render().shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    def observation(self, obs):
        return self.env.render()

def collect_features(model_path, env_name, n_episodes=800, seed=42, is_sac=True):
    print(f"Loading model from {model_path}...")
    if is_sac:
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)

    raw_env_ref = [None]
    def make_env():
        def _init():
            env = gym.make(env_name, render_mode="rgb_array")
            env = PixelObservationWrapper(env)
            env.reset(seed=seed)
            raw_env_ref[0] = env
            return env
        return _init

    env = DummyVecEnv([make_env()])
    env = VecTransposeImage(env)

    features_list = []
    actions_list = []
    obs_pixel_list = []

    print(f"Collecting {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            raw_env = raw_env_ref[0].unwrapped
            
            # For lunar lander, we only have pixel obs
            pixel_obs = raw_env.render()
            obs_pixel_list.append(pixel_obs)

            action, _ = model.predict(obs, deterministic=True)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            
            # SAC uses actor, PPO uses policy
            if is_sac:
                features = model.actor.features_extractor(obs_tensor)
            else:
                features = model.policy.features_extractor(obs_tensor)
                
            features_list.append(features.cpu())
            
            # Ensure actions are 2D floats [batch, action_dim]
            actions_list.append(torch.tensor(action, dtype=torch.float32))
            
            obs, rewards, dones, infos = env.step(action)
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()
        pbar.close()

    env.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    observations_pixel = torch.tensor(np.stack(obs_pixel_list))

    print(f"Collected {len(features)} samples.")
    print(f"  Feature dim: {features.shape}")
    print(f"  Action dim:  {actions.shape}")
    print(f"  Pixel obs:   {observations_pixel.shape}")

    return features, actions, observations_pixel

def main():
    parser = argparse.ArgumentParser(description="Collect Features for Continuous Environments")
    parser.add_argument("--model_path", type=str, default="sac_lunar_lander_continuous.zip")
    parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--algo", type=str, choices=["sac", "ppo"], default="sac", help="Algorithm used")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./collected_data")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    is_sac = (args.algo.lower() == "sac")
    features, actions, obs_pixel = collect_features(
        args.model_path, args.env_name, args.n_episodes, args.seed, is_sac
    )

    save_path = os.path.join(args.save_dir, "continuous_data.pt")
    torch.save({
        "features": features,
        "actions": actions,
        "observations_pixel": obs_pixel,
    }, save_path)
    print(f"Data successfully saved to {save_path}")

if __name__ == "__main__":
    main()