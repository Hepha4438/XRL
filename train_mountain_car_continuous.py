"""
Train SAC with CNN Policy on MountainCarContinuous-v0 (Pixel Observations)
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
    """Wrapper to convert and resize vector observations to GRAYSCALE pixel observations."""
    def __init__(self, env):
        super().__init__(env)
        assert env.render_mode == "rgb_array", "Environment must have render_mode='rgb_array'"
        self.env.reset()
        
        # Đổi shape từ (84, 84, 3) xuống (84, 84, 1) cho Grayscale
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # 1. Lấy frame ảnh RGB gốc
        frame = self.env.render()
        
        # 2. Chuyển sang ảnh xám (Grayscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 3. Resize xuống 84x84
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        
        # 4. Thêm chiều channel: (84, 84) -> (84, 84, 1)
        # Bắt buộc phải có bước này vì Stable Baselines 3 CNN Policy yêu cầu input có dạng (H, W, C)
        frame = np.expand_dims(frame, axis=-1)
        
        return frame
    
def make_env(seed=0):
    def _init():
        # Đổi sang môi trường MountainCarContinuous-v0
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
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
        buffer_size=50000,    
        batch_size=256,
        ent_coef="auto",
        gamma=0.9999,
        tau=0.005,
        use_sde=True,         
        sde_sample_freq=4,
        learning_starts=10000,
        verbose=1,
        tensorboard_log="./tb_mountain_car_continuous/",
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Cấu hình EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_mountain_car_continuous/",
        log_path="./eval_logs_mountain_car_continuous/",
        eval_freq=5000, 
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print("Starting SAC training on pixels (84x84)...")
    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    model.save("sac_mountain_car_continuous", exclude=["replay_buffer"])
    print("Training finished. Model saved without replay buffer.")

if __name__ == "__main__":
    main()