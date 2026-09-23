"""
PPO Training Script for MiniGrid DoorKey 6x6
=============================================
This script trains a PPO agent and ensures that the strictly BEST performing 
model during evaluation is saved as the final output.
"""

import os
import gymnasium as gym
import minigrid  # noqa: F401, needed so envs are registered
import numpy as np
import torch
import torch.nn as nn

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


# ------------------------------------------------------------
# Custom CNN extractor for MiniGrid image observations
# ------------------------------------------------------------
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # After VecTransposeImage, observation_space is in CHW format: (3, 7, 7)
        n_input_channels = observation_space.shape[0]
        print(f"MiniGrid observation shape: {observation_space.shape}, using {n_input_channels} input channels")

        # Deeper architecture for the larger 6x6 grid
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
        # VecTransposeImage ensures observations are already in NCHW format
        return self.linear(self.cnn(observations.float()))


def make_env(render_mode=None, seed=42):
    def _init():
        env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=render_mode)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    seed = 42
    n_envs = 8
    best_model_dir = "./best_doorkey_6x6/"
    final_model_name = "ppo_doorkey_6x6"

    train_env = DummyVecEnv([make_env(seed=seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=1000)])

    # SB3 CNN expects channel-first images
    train_env = VecTransposeImage(train_env)
    eval_env = VecTransposeImage(eval_env)

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_doorkey_6x6/",
        seed=seed,
        device="cuda",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path="./eval_logs_doorkey_6x6/",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    print("\nStarting PPO training...")
    model.learn(total_timesteps=600_000, callback=eval_callback)

    # ------------------------------------------------------------
    # Load the best model evaluated during training to prevent saving 
    # a degraded final policy.
    # ------------------------------------------------------------
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"\n[+] Training completed. Loading the BEST model from: {best_model_path}")
        model = PPO.load(best_model_path, env=train_env)
    else:
        print("\n[!] Warning: Best model not found. Saving the final iteration model instead.")

    print(f"[+] Saving final optimized model as: {final_model_name}.zip")
    model.save(final_model_name)


if __name__ == "__main__":
    main()