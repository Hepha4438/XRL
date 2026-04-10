import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from utils_env import make_env_by_name

def make_env(seed=0):
    def _init():
        return make_env_by_name("PixelCartPole-v0", render_mode="rgb_array", seed=seed)
    return _init

def main():
    seed = 42
    n_envs = 8

    train_env = DummyVecEnv([make_env(seed=seed + i) for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(seed=1000)])

    # SB3 CNN expects channel-first images
    train_env = VecTransposeImage(train_env)
    eval_env = VecTransposeImage(eval_env)

    # Use standard CnnPolicy for 84x84x4 states
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tb_pixel_cartpole/",
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", # automatically uses cuda, mps or cpu
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_pixel_cartpole/",
        log_path="./eval_logs_pixel_cartpole/",
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=400_000, callback=eval_callback)
    model.save("ppo_pixel_cartpole")

if __name__ == "__main__":
    main()
