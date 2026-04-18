#!/usr/bin/env python3
"""Test if the Pong model can be loaded and used."""

import sys
sys.path.insert(0, '/Users/HIEU/Workspaces/Projects/XRL')

try:
    from stable_baselines3 import PPO
    print("✓ stable_baselines3 imported successfully")
    
    # Try to load the Pong model
    model = PPO.load("ppo-PongNoFrameskip-v4.zip", device="cpu")
    print("✓ Model loaded successfully!")
    print(f"  Policy: {model.policy.__class__.__name__}")
    print(f"  Features extractor: {model.policy.features_extractor.__class__.__name__}")
    print(f"  Observation space: {model.observation_space}")
    print(f"  Action space: {model.action_space}")
    
    # Test prediction
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStack
    
    env = gym.make("ALE/Pong-v5", render_mode=None)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, terminal_on_life_loss=False)
    env = FrameStack(env, num_stack=4)
    
    obs, info = env.reset()
    print(f"✓ Environment created. Observation shape: {obs.shape}")
    
    action, _states = model.predict(obs, deterministic=False)
    print(f"✓ Prediction works! Action: {action}")
    print("\n✅ Model is fully compatible!")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
