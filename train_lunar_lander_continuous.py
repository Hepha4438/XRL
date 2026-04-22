"""
Train SAC with CNN Policy on LunarLanderContinuous-v3 (Pixel Observations)
"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class PixelObservationWrapper(gym.ObservationWrapper):
    """Wrapper to convert vector observations to pixel observations."""
    def __init__(self, env):
        super().__init__(env)
        # Ensure the environment was created with render_mode="rgb_array"
        assert env.render_mode == "rgb_array", "Environment must have render_mode='rgb_array'"
        self.env.reset()
        # Get a sample frame to determine shape
        sample_image = self.env.render()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=sample_image.shape, dtype=np.uint8
        )

    def observation(self, obs):
        return self.env.render()

def make_env(seed=0):
    def _init():
        env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        env = PixelObservationWrapper(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return _init

def main():
    seed = 42
    n_envs = 4 # SAC is generally more sample efficient but slower per step than PPO

    train_env = DummyVecEnv([make_env(seed=seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=1000)])

    # SB3 CNN expects channel-first images (C, H, W)
    train_env = VecTransposeImage(train_env)
    eval_env = VecTransposeImage(eval_env)

    # Use standard CnnPolicy for continuous SAC
    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        verbose=1,
        tensorboard_log="./tb_lunar_lander_continuous/",
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_lunar_lander_continuous/",
        log_path="./eval_logs_lunar_lander_continuous/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print("Starting SAC training on pixels...")
    model.learn(total_timesteps=500_000, callback=eval_callback)
    model.save("sac_lunar_lander_continuous")

if __name__ == "__main__":
    main()