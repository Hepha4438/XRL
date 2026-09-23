#!/usr/bin/env python3
"""
collect_with_observations.py
Collect held-out PPO rollouts (features + teacher actions + observations) for
MiniGrid, PixelCartPole and Atari (PongNoFrameskip-v4, BoxingNoFrameskip-v4).

Features are extracted EXACTLY as in feature_collect.py (the source of LUCID's
training data), i.e. through SB3's own preprocessing:

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    features = model.policy.extract_features(obs_tensor, model.policy.features_extractor)

`extract_features` calls `preprocess_obs`, which divides uint8 image observations
by 255 (policy.normalize_images=True). Calling `features_extractor(obs)` directly
on the raw uint8 tensor -- as the previous version of this script did -- skips that
step and yields features from a different distribution than the training data.
On the first step the script cross-checks this against an explicit `/ 255.0`
(Method 3 of test_obs_255.py) and aborts if the two disagree.

Environments are built the same way as in feature_collect.py:
  * Atari   : make_atari_env(env_id, n_envs=1, seed) + VecFrameStack(4) + VecTransposeImage
  * MiniGrid: gym.make(..., render_mode="rgb_array") + ImgObsWrapper, DummyVecEnv + VecTransposeImage
  * PixelCartPole: utils_env.make_env_by_name (4-frame stack), DummyVecEnv + VecTransposeImage

IMPORTANT: use a --seed different from the one used for the training data
(feature_collect.py defaults to seed 0), otherwise the "held-out" set repeats
the training episodes.

Usage:
  # Atari (observation = stacked 4x84x84 model input; add --save_obs_mode both for RGB frames)
  python collect_with_observations.py \
      --model_path pong_sb3_290.zip --env_name PongNoFrameskip-v4 \
      --n_episodes 20 --seed 1000 --save_path held_out_pong_data.pt

  python collect_with_observations.py \
      --model_path boxing_sb3_290.zip --env_name BoxingNoFrameskip-v4 \
      --n_episodes 20 --seed 1000 --save_path held_out_boxing_data.pt

  # MiniGrid / PixelCartPole (unchanged interface)
  python collect_with_observations.py \
      --model_path ppo_doorkey_6x6.zip --env_name MiniGrid-DoorKey-6x6-v0 \
      --n_episodes 2000 --save_path held_out_doorkey_data.pt --seed 42

Saved dict keys:
  features            (N, d) float32   -- SB3-preprocessed CNN features
  actions             (N,)   int64     -- deterministic teacher actions
  observations        grid encoding (MiniGrid, (N,7,7,3)) or model input (Atari/CartPole, (N,C,84,84)) uint8
  observations_pixel  rendered RGB frames (optional, see --save_obs_mode)
  episode_ids, step_ids (N,) int64     -- for episode-level (cluster) statistics
  rewards, dones      (N,)
  meta                dict (env_name, seed, feature_extraction, ...)
"""

import argparse
import os
import sys

import numpy as np
import torch
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)          # required by gymnasium >= 1.0 for ALE ids
except Exception:
    pass
try:
    import minigrid  # noqa: F401  (registers MiniGrid envs)
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

CANONICAL_ATARI = {"pong": "PongNoFrameskip-v4", "boxing": "BoxingNoFrameskip-v4"}


# ============================================================================
# Environment
# ============================================================================

def env_kind(env_name):
    if "MiniGrid" in env_name:
        return "minigrid"
    if "NoFrameskip" in env_name or "Pong" in env_name or "Boxing" in env_name or env_name.startswith("ALE/"):
        return "atari"
    return "other"


def canonical_env_name(env_name):
    """The PPO teachers in this repo were trained on *NoFrameskip-v4 (no sticky actions)."""
    low = env_name.lower()
    for key, canon in CANONICAL_ATARI.items():
        if key in low and env_name != canon:
            print(f"[WARN] '{env_name}' -> using '{canon}' (the id the teacher was trained on; "
                  f"*NoFrameskip-v0 has sticky actions p=0.25).")
            return canon
    return env_name


def make_env(env_name, seed, need_rgb):
    """Returns (vec_env, raw_env_holder). raw_env_holder[0] is the unwrapped single env (non-Atari)."""
    kind = env_kind(env_name)
    raw = [None]

    if kind == "atari":
        env_kwargs = {"render_mode": "rgb_array"} if need_rgb else None
        venv = make_atari_env(env_name, n_envs=1, seed=seed, env_kwargs=env_kwargs)
        venv = VecFrameStack(venv, n_stack=4)
        venv = VecTransposeImage(venv)
        return venv, raw

    if kind == "minigrid":
        from minigrid.wrappers import ImgObsWrapper

        def _init():
            e = gym.make(env_name, render_mode="rgb_array")
            raw[0] = e
            e = ImgObsWrapper(e)
            e.reset(seed=seed)
            return e
    else:
        from utils_env import make_env_by_name

        def _init():
            e = make_env_by_name(env_name, render_mode="rgb_array", seed=seed)
            raw[0] = e
            return e

    venv = DummyVecEnv([_init])
    venv = VecTransposeImage(venv)
    venv.seed(seed)
    return venv, raw


# ============================================================================
# Feature extraction (identical to feature_collect.py)
# ============================================================================

@torch.no_grad()
def extract_features(model, obs):
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    feats = model.policy.extract_features(obs_tensor, model.policy.features_extractor)
    if isinstance(feats, tuple):          # share_features_extractor=False -> (pi, vf)
        feats = feats[0]
    return feats, obs_tensor


@torch.no_grad()
def check_normalization(model, obs, feats, atol=1e-4):
    """Cross-check SB3 preprocessing against an explicit /255 (test_obs_255.py, Method 3)."""
    obs_space = model.policy.observation_space
    image = is_image_space(obs_space)
    normalize = bool(getattr(model.policy, "normalize_images", True))
    raw_max = float(np.max(obs))
    print(f"[check] raw obs max = {raw_max:.1f} | is_image_space = {image} | normalize_images = {normalize}")
    if not (image and normalize):
        print("[check] SB3 does not rescale this observation space; features use the raw observation "
              "(same as the training data from feature_collect.py).")
        return
    x = torch.as_tensor(obs).float().to(model.device) / 255.0
    f3 = model.policy.features_extractor(x)
    err = float((f3 - feats).abs().max())
    print(f"[check] max |features(SB3 preprocess) - features(obs/255)| = {err:.2e}")
    if err > atol:
        raise RuntimeError("Feature extraction does not match the explicit /255 path -- aborting "
                           "(held-out features would not be comparable with training features).")
    raw_feats = model.policy.features_extractor(torch.as_tensor(obs).float().to(model.device))
    print(f"[check] (for reference) without /255 the features would differ by "
          f"{float((raw_feats - feats).abs().mean()):.3f} on average -> old bug avoided.")


# ============================================================================
# Observations
# ============================================================================

def grab_observations(kind, vec_env, raw_env, obs, mode, tile_size):
    grid, pixel = None, None
    want_grid = mode in ("grid", "both")
    want_pixel = mode in ("pixel", "both")

    if kind == "minigrid":
        u = raw_env.unwrapped
        g = u.gen_obs()["image"]
        if want_grid:
            grid = g.copy()
        if want_pixel:
            try:
                pixel = u.get_obs_render(g, tile_size=tile_size)
            except Exception:
                pixel = u.render()
    else:
        if want_grid:                                   # exact model input (C, 84, 84) uint8
            grid = np.asarray(obs[0]).astype(np.uint8).copy()
        if want_pixel:
            try:
                if kind == "atari":
                    pixel = vec_env.get_images()[0]
                else:
                    pixel = raw_env.render()
            except Exception:
                pixel = None
    return grid, pixel


# ============================================================================
# Collection
# ============================================================================

def collect_with_observations(model_path, env_name, n_episodes=800, seed=42, tile_size=8,
                              save_obs_mode="auto", max_total_steps=0, device="auto"):
    env_name = canonical_env_name(env_name)
    kind = env_kind(env_name)
    if save_obs_mode == "auto":
        save_obs_mode = "both" if kind == "minigrid" else "grid"
    need_rgb = save_obs_mode in ("pixel", "both")

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    print(f"Loading PPO model from {model_path} (device={dev})...")
    model = PPO.load(model_path, device=dev)

    vec_env, raw = make_env(env_name, seed, need_rgb)
    if vec_env.observation_space.shape != model.observation_space.shape:
        raise RuntimeError(f"Env observation shape {vec_env.observation_space.shape} != model "
                           f"observation shape {model.observation_space.shape}. Wrong env wrappers/id?")
    print(f"Env: {env_name} [{kind}] | obs {vec_env.observation_space.shape} | "
          f"actions {vec_env.action_space} | save_obs_mode={save_obs_mode} | seed={seed}")

    feats_l, acts_l, grid_l, pix_l = [], [], [], []
    ep_l, step_l, rew_l, done_l = [], [], [], []

    obs = vec_env.reset()
    episode, t_in_ep, total = 0, 0, 0
    checked = False
    pbar = tqdm(total=n_episodes, desc="Episodes")
    while episode < n_episodes:
        grid, pixel = grab_observations(kind, vec_env, raw[0], obs, save_obs_mode, tile_size)
        if grid is not None:
            grid_l.append(grid)
        if pixel is not None:
            pix_l.append(pixel)

        action, _ = model.predict(obs, deterministic=True)
        feats, _ = extract_features(model, obs)
        if not checked:
            check_normalization(model, obs, feats)
            checked = True

        feats_l.append(feats.float().cpu())
        acts_l.append(torch.as_tensor(action).long().view(-1))
        ep_l.append(episode)
        step_l.append(t_in_ep)

        obs, rewards, dones, infos = vec_env.step(action)
        rew_l.append(float(rewards[0]))
        done_l.append(bool(dones[0]))
        total += 1
        t_in_ep += 1

        # Atari: EpisodicLifeEnv signals `done` on life loss; the Monitor's "episode"
        # key marks the real end of a game. Other envs: `done` is the episode end.
        if dones[0]:
            real_end = ("episode" in infos[0]) if kind == "atari" else True
            if real_end:
                episode += 1
                t_in_ep = 0
                pbar.update(1)
        if max_total_steps and total >= max_total_steps:
            print(f"\nReached --max_total_steps={max_total_steps}; stopping after {episode} full episodes.")
            break
    pbar.close()
    vec_env.close()

    result = {
        "features": torch.cat(feats_l, 0),
        "actions": torch.cat(acts_l, 0),
        "episode_ids": torch.tensor(ep_l, dtype=torch.long),
        "step_ids": torch.tensor(step_l, dtype=torch.long),
        "rewards": torch.tensor(rew_l, dtype=torch.float32),
        "dones": torch.tensor(done_l, dtype=torch.bool),
        "meta": {
            "env_name": env_name, "seed": seed, "n_episodes": episode, "model_path": model_path,
            "feature_extraction": "policy.extract_features(obs_to_tensor(obs)) [SB3 preprocess, /255 for images]",
            "save_obs_mode": save_obs_mode,
        },
    }
    if grid_l:
        result["observations"] = torch.from_numpy(np.stack(grid_l))
    if pix_l:
        try:
            result["observations_pixel"] = torch.from_numpy(np.stack(pix_l))
        except ValueError:
            print("[WARN] Rendered frames have inconsistent shapes; observations_pixel not saved.")

    print(f"  Features: {tuple(result['features'].shape)} | Actions: {tuple(result['actions'].shape)} "
          f"| Steps: {total} | Episodes: {episode}")
    if "observations" in result:
        print(f"  observations: {tuple(result['observations'].shape)} {result['observations'].dtype}")
    if "observations_pixel" in result:
        print(f"  observations_pixel: {tuple(result['observations_pixel'].shape)}")
    counts = torch.bincount(result["actions"])
    print(f"  Teacher action counts: {counts.tolist()}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Collect held-out PPO rollouts with observations")
    ap.add_argument("--model_path", type=str, default="ppo_doorkey_6x6.zip")
    ap.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0",
                    help="MiniGrid-*, PixelCartPole-v0, PongNoFrameskip-v4, BoxingNoFrameskip-v4")
    ap.add_argument("--n_episodes", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42,
                    help="Use a seed different from the training rollouts (feature_collect.py default: 0)")
    ap.add_argument("--tile_size", type=int, default=8, help="MiniGrid render tile size")
    ap.add_argument("--save_obs_mode", type=str, default="auto",
                    choices=["auto", "none", "grid", "pixel", "both"],
                    help="auto: MiniGrid=both, Atari/CartPole=grid (model input frames). "
                         "Atari RGB frames (pixel/both) cost ~100 KB/step.")
    ap.add_argument("--max_total_steps", type=int, default=0, help="Safety cap on total steps (0 = none)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--save_path", type=str, default="./stage1_outputs/collected_data_with_obs.pt")
    args = ap.parse_args()

    data = collect_with_observations(args.model_path, args.env_name, args.n_episodes, args.seed,
                                     args.tile_size, args.save_obs_mode, args.max_total_steps, args.device)
    d = os.path.dirname(args.save_path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(data, args.save_path)
    print(f"\nSaved to {args.save_path} ({os.path.getsize(args.save_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
