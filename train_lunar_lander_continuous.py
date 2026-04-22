"""
Train SAC with CNN Policy on LunarLanderContinuous-v3 (Pixel Observations)
Optimized for Kaggle: Reduced file size and memory usage.
"""
import gymnasium as gym
import numpy as np
import torch
import cv2
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

class PixelObservationWrapper(gym.ObservationWrapper):
    """Wrapper to convert and resize vector observations to pixel observations."""
    def __init__(self, env):
        super().__init__(env)
        assert env.render_mode == "rgb_array", "Environment must have render_mode='rgb_array'"
        self.env.reset()
        # Resize xuống 84x84 là chuẩn chung cho RL từ hình ảnh (Atari/DeepMind)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

    def observation(self, obs):
        # Lấy frame và resize ngay lập tức để tiết kiệm bộ nhớ Buffer
        frame = self.env.render()
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame

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
    n_envs = 4 

    train_env = DummyVecEnv([make_env(seed=seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=1000)])

    train_env = VecTransposeImage(train_env)
    eval_env = VecTransposeImage(eval_env)

    model = SAC(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=50000, # Với ảnh 84x84, 50k bước sẽ chiếm khoảng ~1GB RAM
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        verbose=1,
        tensorboard_log="./tb_lunar_lander_continuous/",
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Cấu hình EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_lunar_lander_continuous/",
        log_path="./eval_logs_lunar_lander_continuous/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print("Starting SAC training on pixels (84x84)...")
    model.learn(total_timesteps=500_000, callback=eval_callback)

    # QUAN TRỌNG: Lưu model cuối cùng và loại bỏ Replay Buffer
    # Điều này sẽ biến file 3GB thành file ~15MB.
    model.save("sac_lunar_lander_continuous", exclude=["replay_buffer"])
    print("Training finished. Model saved without replay buffer.")

if __name__ == "__main__":
    main()