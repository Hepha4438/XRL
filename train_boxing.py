import argparse, ale_py, gymnasium as gym
gym.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    VecFrameStack, VecTransposeImage, SubprocVecEnv)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "BoxingNoFrameskip-v4"

def lin(x0):
    return lambda p: p * x0

def make(n_envs, seed):
    e = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed,
                       vec_env_cls=SubprocVecEnv if n_envs > 1 else None)
    return VecTransposeImage(VecFrameStack(e, n_stack=4))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/boxing_s0")
    ap.add_argument("--const-lr", action="store_true")
    a = ap.parse_args()

    env = make(a.n_envs, a.seed)
    eval_env = make(1, a.seed + 1000)

    model = PPO(
        "CnnPolicy", env,
        n_steps=128, batch_size=256, n_epochs=4,
        learning_rate=2.5e-4 if a.const_lr else lin(2.5e-4), clip_range=0.1 if a.const_lr else lin(0.1),
        ent_coef=0.01, vf_coef=0.5,
        gamma=0.99, gae_lambda=0.95,
        verbose=1, seed=a.seed, device="cuda",
        tensorboard_log=f"{a.out}/tb",
    )

    model.learn(
        total_timesteps=a.timesteps,
        callback=EvalCallback(eval_env, best_model_save_path=f"{a.out}/best",
                              log_path=f"{a.out}/eval", eval_freq=max(50_000 // a.n_envs, 1),
                              n_eval_episodes=5, deterministic=True),
    )
    model.save(f"{a.out}/final_model")

    for tag, path in [("final", f"{a.out}/final_model"),
                      ("best", f"{a.out}/best/best_model")]:
        try:
            m = PPO.load(path, env=eval_env, device="cuda")
            for det in (True, False):
                print(tag, "det" if det else "stoch",
                      evaluate_policy(m, eval_env, n_eval_episodes=10,
                                      deterministic=det))
        except Exception as e:
            print(tag, "skip:", e)
