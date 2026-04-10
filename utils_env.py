import gymnasium as gym
import collections
import cv2
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from minigrid.wrappers import ImgObsWrapper

class PixelCartPoleWrapper(gym.ObservationWrapper):
    def __init__(self, render_mode="rgb_array"):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, _):
        obs = self.env.render()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, -1)

class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        # Determine the stacked shape
        old_shape = self.observation_space.shape
        new_shape = old_shape[:-1] + (old_shape[-1] * num_stack,)
        
        low = np.repeat(self.observation_space.low, num_stack, axis=-1)
        high = np.repeat(self.observation_space.high, num_stack, axis=-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=new_shape, dtype=self.observation_space.dtype)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

def make_env_by_name(env_name, render_mode="rgb_array", seed=0):
    """Unified environment maker for MiniGrid, Atari, and PixelCartPole"""
    if "PixelCartPole" in env_name:
        # pixel wrapper always needs rendering enabled
        env = PixelCartPoleWrapper(render_mode="rgb_array")
        env = FrameStack(env, num_stack=4)
        env.reset(seed=seed)
        return env
    else:
        env = gym.make(env_name, render_mode=render_mode)
        if "MiniGrid" in env_name:
            env = ImgObsWrapper(env)
        elif "ALE/" in env_name:
            env = AtariWrapper(env, clip_reward=False)
            env = FrameStack(env, num_stack=4)
        env.reset(seed=seed)
        return env
