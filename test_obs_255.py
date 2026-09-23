"""
test_obs_255.py
Quick script to verify specific feature extraction snippets.
"""

import argparse
import torch
import numpy as np

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to PPO model")
    parser.add_argument("--env_name", type=str, required=True, help="Environment name")
    return parser.parse_args()

def make_test_env(env_name: str):
    is_atari = "NoFrameskip" in env_name or "Boxing" in env_name or "Pong" in env_name
    is_minigrid = "MiniGrid" in env_name

    if is_atari:
        env = make_atari_env(env_name, n_envs=1, seed=42, wrapper_kwargs={"clip_reward": False})
        env = VecFrameStack(env, n_stack=4)
        return VecTransposeImage(env)
    elif is_minigrid:
        import minigrid
        from minigrid.wrappers import ImgObsWrapper
        def _init():
            e = gym.make(env_name, render_mode="rgb_array")
            e = ImgObsWrapper(e)
            e.reset(seed=42)
            return e
        env = DummyVecEnv([_init])
        return VecTransposeImage(env)
    else:
        # Fallback
        def _init():
            e = gym.make(env_name)
            e.reset(seed=42)
            return e
        env = DummyVecEnv([_init])
        if len(env.observation_space.shape) == 3:
            env = VecTransposeImage(env)
        return env

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Model: {args.model_path}")
    model = PPO.load(args.model_path, device=device)
    
    print(f"Loading Environment: {args.env_name}")
    env = make_test_env(args.env_name)
    obs = env.reset()

    print("\n" + "="*70)
    print("TESTING OBSERVATION PREPROCESSING & FEATURE EXTRACTION")
    print("="*70)

    # ---------------------------------------------------------
    # METHOD 1: GROUND TRUTH (How SB3 Teacher internally calls it)
    # ---------------------------------------------------------
    obs_tensor_true, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        features_true = model.policy.extract_features(obs_tensor_true)
        
    print("--- METHOD 1: GROUND TRUTH (SB3 Internal expected call) ---")
    print(f"Raw obs array max val  : {obs.max()}")
    print(f"Features mean (True)   : {features_true.mean().item():.4f}")
    print(f"Features std  (True)   : {features_true.std().item():.4f}")

    # ---------------------------------------------------------
    # METHOD 2: THE EXACT SNIPPET YOU WANT TO TEST
    # ---------------------------------------------------------
    print("\n--- METHOD 2: EXACT SNIPPET YOU REQUESTED ---")
    try:
        # 1. Obs to tensor
        obs_tensor_test, _ = model.policy.obs_to_tensor(obs)
        
        with torch.no_grad():
            # 2. Extract features exactly as requested
            features_test = model.policy.extract_features(obs_tensor_test, model.policy.features_extractor)
            
        print(f"Input tensor max val   : {obs_tensor_test.max().item()}")
        print(f"Features mean (Test)   : {features_test.mean().item():.4f}")
        print(f"Features std  (Test)   : {features_test.std().item():.4f}")
        
        error_test = torch.abs(features_true - features_test).mean().item()
        print(f">>> L1 Error vs True   : {error_test:.4f}")
        
    except Exception as e:
        print(f"[!] CRASH DETECTED")
        print(f"Error Type: {type(e).__name__}")
        print(f"Message   : {e}")

    # ---------------------------------------------------------
    # METHOD 3: EXPLICIT / 255.0
    # ---------------------------------------------------------
    obs_tensor_fixed = torch.as_tensor(obs).float().to(device) / 255.0
    with torch.no_grad():
        features_fixed = model.policy.features_extractor(obs_tensor_fixed)

    print("\n--- METHOD 3: EXPLICIT / 255.0 ---")
    print(f"Input tensor max val   : {obs_tensor_fixed.max().item():.4f}")
    print(f"Features mean (Fixed)  : {features_fixed.mean().item():.4f}")
    print(f"Features std  (Fixed)  : {features_fixed.std().item():.4f}")
    
    error_fixed = torch.abs(features_true - features_fixed).mean().item()
    print(f">>> L1 Error vs True   : {error_fixed:.4f}")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()