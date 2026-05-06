"""
Stage 1: Feature Collect
================================
Outputs:
    - Feature normalization stats (mean, std)
    - Diagnostic plots and report

Usage:
    python feature_collect.py --features_path ./collected_data/features.pt
    python feature_collect.py --model_path ppo_doorkey_6x6.zip --env_name MiniGrid-DoorKey-6x6-v0
"""

import argparse
import os
import json
import warnings

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Feature normalization stats
# ---------------------------------------------------------------------------

def compute_normalization_stats(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, 1e-6)

    print(f"\n{'='*60}")
    print(f"NORMALIZATION STATS")
    print(f"{'='*60}")
    print(f"  Feature dimension          : {X.shape[1]}")
    print(f"  Mean range                 : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range                  : [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Near-zero std dims (<1e-4) : {(std < 1e-4).sum()}")
    print(f"  High-variance dims (>10)   : {(std > 10).sum()}")

    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Action distribution analysis
# ---------------------------------------------------------------------------

def action_distribution_analysis(actions: np.ndarray, n_actions: int = 7):
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    counts = np.bincount(actions, minlength=n_actions)
    freqs = counts / counts.sum()

    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Total samples: {len(actions)}")
    for a in range(n_actions):
        bar = "█" * int(freqs[a] * 50)
        print(f"  {action_names[a]:10s}: {counts[a]:6d} ({freqs[a]*100:5.1f}%) {bar}")

    freqs_nonzero = freqs[freqs > 0]
    entropy = -np.sum(freqs_nonzero * np.log2(freqs_nonzero))
    max_entropy = np.log2(n_actions)
    print(f"\n  Entropy: {entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  Normalized entropy: {entropy/max_entropy:.3f}")

    return {"counts": counts, "freqs": freqs, "entropy": entropy}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diagnostics(norm_stats, action_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Stage 1: Feature Space Analysis", fontsize=14, fontweight="bold")

    std = norm_stats["std"]
    ax.hist(std, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Per-dimension std"); ax.set_ylabel("Count")
    ax.set_title("Feature Std Distribution"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "stage1_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Diagnostic plot saved: {plot_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    freqs = action_stats["freqs"]
    bars = ax2.bar(action_names, freqs * 100, color="steelblue", alpha=0.8, edgecolor="black")
    ax2.set_ylabel("Frequency (%)"); ax2.set_title("Action Distribution in Rollout Data")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, f in zip(bars, freqs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{f*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    action_plot_path = os.path.join(save_dir, "action_distribution.png")
    plt.savefig(action_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Action distribution plot saved: {action_plot_path}")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_stage1_outputs(norm_stats, action_stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    stage1_data = {
        "feature_mean": torch.from_numpy(norm_stats["mean"]).float(),
        "feature_std": torch.from_numpy(norm_stats["std"]).float(),
        "action_counts": torch.from_numpy(action_stats["counts"]).long(),
        "action_freqs": torch.from_numpy(action_stats["freqs"]).float(),
        "action_entropy": action_stats["entropy"],
    }

    save_path = os.path.join(save_dir, "stage1_outputs.pt")
    torch.save(stage1_data, save_path)
    print(f"\n  Stage 1 outputs saved: {save_path}")

    summary = {
        "feature_mean_range": [float(norm_stats["mean"].min()), float(norm_stats["mean"].max())],
        "feature_std_range": [float(norm_stats["std"].min()), float(norm_stats["std"].max())],
        "action_entropy": float(action_stats["entropy"]),
    }

    summary_path = os.path.join(save_dir, "stage1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Stage 1 summary saved: {summary_path}")

    return save_path


def load_stage1_outputs(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Loaded Stage 1 outputs from {path}")
    return data


# ---------------------------------------------------------------------------
# Data collection — NOW WITH OBSERVATIONS
# ---------------------------------------------------------------------------

def collect_features(model_path, env_name, n_episodes=800, seed=42, tile_size=8):
    """Collect features, actions, grid obs, AND pixel-rendered observations."""
    import gymnasium as gym
    import minigrid  # noqa: F401
    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from tqdm import tqdm
    
    # Register ALE (Atari) environments if needed
    try:
        import ale_py  # noqa: F401
    except Exception:
        pass
    
    # Simple frame stack wrapper for Atari
    class FrameStack(gym.Wrapper):
        def __init__(self, env, num_stack=4):
            super().__init__(env)
            self.num_stack = num_stack
            self.frames = collections.deque(maxlen=num_stack)
            # Update observation space for stacked frames
            shape = env.observation_space.shape
            if len(shape) == 3:  # HWC
                self.observation_space = gym.spaces.Box(
                    low=env.observation_space.low.min(),
                    high=env.observation_space.high.max(),
                    shape=(shape[0], shape[1], shape[2] * num_stack),
                    dtype=env.observation_space.dtype
                )
            else:  # Already grayscale or other
                self.observation_space = gym.spaces.Box(
                    low=env.observation_space.low.min(),
                    high=env.observation_space.high.max(),
                    shape=(*shape, num_stack),
                    dtype=env.observation_space.dtype
                )
        
        def _get_obs(self):
            assert len(self.frames) == self.num_stack
            frames_list = list(self.frames)
            # Stack along last axis
            stacked = np.concatenate(frames_list, axis=-1)
            return stacked
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            for _ in range(self.num_stack):
                self.frames.append(obs)
            return self._get_obs(), info
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_obs(), reward, terminated, truncated, info
    
    import collections

    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)

    raw_env_ref = [None]

    def make_env():
        def _init():
            from utils_env import make_env_by_name
            env = make_env_by_name(env_name, render_mode="rgb_array", seed=seed)
            raw_env_ref[0] = env
            return env
        return _init

    env = DummyVecEnv([make_env()])
    # Apply VecTransposeImage for both MiniGrid and Atari (they output HWC format)
    env = VecTransposeImage(env)

    features_list = []
    actions_list = []
    obs_grid_list = []
    obs_pixel_list = []

    print(f"Collecting {n_episodes} episodes...")
    obs = env.reset()
    episode_count = 0

    with torch.no_grad():
        pbar = tqdm(total=n_episodes)
        while episode_count < n_episodes:
            raw_env = raw_env_ref[0].unwrapped
            try:
                # Grid encoding (for dedup / hashing)
                grid_obs = raw_env.gen_obs()['image']
                obs_grid_list.append(grid_obs.copy())

                # Pixel render via MiniGrid's own renderer
                # This handles orientation, agent marker, colors correctly
                pixel_obs = raw_env.get_obs_render(grid_obs, tile_size=tile_size)
                obs_pixel_list.append(pixel_obs)
            except Exception:
                obs_grid_list.append(np.zeros((7, 7, 3), dtype=np.uint8))
                obs_pixel_list.append(
                    np.zeros((7 * tile_size, 7 * tile_size, 3), dtype=np.uint8))

            action, _ = model.predict(obs, deterministic=True)
            obs_tensor = torch.as_tensor(obs).float().to(model.device)
            features = model.policy.features_extractor(obs_tensor)
            features_list.append(features.cpu())
            actions_list.append(torch.tensor(action))
            obs, rewards, dones, infos = env.step(action)
            if dones[0]:
                episode_count += 1
                pbar.update(1)
                obs = env.reset()
        pbar.close()

    env.close()

    features = torch.cat(features_list, dim=0)
    actions = torch.cat(actions_list, dim=0)
    observations = torch.tensor(np.stack(obs_grid_list))
    observations_pixel = torch.tensor(np.stack(obs_pixel_list))

    print(f"Collected {len(features)} samples, feature dim = {features.shape[1]}")
    print(f"  Grid observations: {observations.shape}")
    print(f"  Pixel observations: {observations_pixel.shape}")

    return features, actions, observations, observations_pixel


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage1(features: torch.Tensor, actions: torch.Tensor,
               save_dir: str = "./stage1_outputs"):
    X = features.numpy()
    A = actions.numpy().astype(int)

    print(f"\n{'#'*60}")
    print(f"  STAGE 1: FEATURE SPACE ANALYSIS")
    print(f"  N = {X.shape[0]}, d = {X.shape[1]}")
    print(f"{'#'*60}")

    norm_stats = compute_normalization_stats(X)
    action_stats = action_distribution_analysis(A)
    plot_diagnostics(norm_stats, action_stats, save_dir)
    save_path = save_stage1_outputs(norm_stats, action_stats, save_dir)

    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR STAGE 2 (SAE TRAINING)")
    print(f"{'='*60}")
    print(f"  → Normalize features with saved mean/std before SAE training")

    return load_stage1_outputs(save_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Feature Space Analysis (No SVD/ICA)")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Path to pre-collected features .pt file (with 'features' and 'actions' keys)")
    parser.add_argument("--model_path", type=str, default="ppo_doorkey_6x6.zip",
                        help="PPO model path (used if --features_path not given)")
    parser.add_argument("--env_name", type=str, default="MiniGrid-DoorKey-6x6-v0",
                        help="Environment name (used if --features_path not given)")
    parser.add_argument("--n_episodes", type=int, default=800,
                        help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./stage1_outputs")

    args = parser.parse_args()

    if args.features_path is not None:
        print(f"Loading features from {args.features_path}...")
        data = torch.load(args.features_path, map_location="cpu", weights_only=False)
        features = data["features"]
        actions = data["actions"]
    else:
        features, actions, observations, observations_pixel = collect_features(
            args.model_path, args.env_name, args.n_episodes, args.seed
        )
        os.makedirs(args.save_dir, exist_ok=True)
        raw_path = os.path.join(args.save_dir, "collected_data.pt")
        torch.save({
            "features": features,
            "actions": actions,
            "observations": observations,            # grid encoding (for dedup)
            "observations_pixel": observations_pixel, # MiniGrid rendered (for display)
        }, raw_path)
        print(f"Raw data saved: {raw_path}")
        print(f"  Grid obs: {observations.shape}")
        print(f"  Pixel obs: {observations_pixel.shape}")

    stage1_data = run_stage1(
        features, actions,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()