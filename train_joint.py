"""
SAE + Product T-Norm Neural Logic Training (JOINT TRAINING BASELINE)
=============================================================
This is an ablation study baseline demonstrating the "Gradient War" 
phenomenon where the SAE and logic layer are trained simultaneously 
from scratch.

Usage:
    python train_joint.py \
        --features_path ./stage1_outputs/collected_data.pt \
        --hidden_dim 300 --k 50 \
        --n_clauses_per_action 10 \
        --n_epochs 400 \
        --save_dir ./sae_logic_joint_outputs
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparse_concept_autoencoder import OvercompleteSAE, SAEConfig, init_from_stage1


# ============================================================================
# Sigmoid Bottleneck
# ============================================================================

class SigmoidBottleneck(nn.Module):
    def __init__(self, n_features: int, initial_alpha: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.full((n_features,), np.log(np.exp(initial_alpha) - 1.0))
        )
        self.beta = nn.Parameter(torch.zeros(n_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.log_alpha) + 0.5
        return torch.sigmoid(alpha * (z - self.beta))

    def get_sharpness(self) -> torch.Tensor:
        return F.softplus(self.log_alpha) + 0.5


# ============================================================================
# Product T-Norm Logic Layer
# ============================================================================

class ProductTNormLogicLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        n_clauses_per_action: int = 10,
        l0_penalty_weight: float = 1e-4,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_clauses_per_action = n_clauses_per_action
        self.l0_penalty_weight = l0_penalty_weight
        total_clauses = n_actions * n_clauses_per_action

        self.w_pos = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.w_neg = nn.Parameter(torch.randn(total_clauses, n_features) * 0.01 - 3.0)
        self.clause_weight = nn.Parameter(torch.ones(total_clauses) * 2.0)

    def _get_selection_probs(self):
        absent_logit = torch.zeros_like(self.w_pos)
        logits = torch.stack([self.w_pos, self.w_neg, absent_logit], dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs[..., 0], probs[..., 1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        p, n = self._get_selection_probs()

        f = features.unsqueeze(1)
        p_ex = p.unsqueeze(0)
        n_ex = n.unsqueeze(0)

        literals = p_ex * f + n_ex * (1.0 - f) + (1.0 - p_ex - n_ex)
        log_literals = torch.log(literals + 1e-8)
        log_clause_sum = log_literals.sum(dim=-1)

        clauses = torch.sigmoid(log_clause_sum + self.clause_weight.unsqueeze(0))
        clauses = clauses.view(batch_size, self.n_actions, self.n_clauses_per_action)
        return clauses.sum(dim=-1)

    def complexity_penalty(self) -> torch.Tensor:
        p, n = self._get_selection_probs()
        return self.l0_penalty_weight * (p + n).mean()

    def extract_rules(self, feature_names=None, action_names=None, threshold=0.3):
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(self.n_features)]
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]

        p, n = self._get_selection_probs()
        p, n = p.detach().cpu().numpy(), n.detach().cpu().numpy()
        cb = self.clause_weight.detach().cpu().numpy()

        rules = {}
        for a in range(self.n_actions):
            clauses = []
            for c in range(self.n_clauses_per_action):
                idx = a * self.n_clauses_per_action + c
                lits = []
                for i in range(self.n_features):
                    if p[idx, i] > threshold:
                        lits.append(f"{feature_names[i]}")
                    elif n[idx, i] > threshold:
                        lits.append(f"¬{feature_names[i]}")
                if lits:
                    clauses.append(f"({' ∧ '.join(lits)}) [bias={cb[idx]:.2f}]")
            rules[action_names[a]] = clauses if clauses else ["(no active clauses)"]
        return rules

    def count_active_rules(self, threshold=0.3, action_names=None):
        if action_names is None:
            action_names = [f"action_{a}" for a in range(self.n_actions)]
        p, n = self._get_selection_probs()
        p, n = p.detach().cpu().numpy(), n.detach().cpu().numpy()

        total_clauses = non_empty = total_literals = 0
        per_action = {}
        for a in range(self.n_actions):
            ac = 0
            for c in range(self.n_clauses_per_action):
                idx = a * self.n_clauses_per_action + c
                nl = ((p[idx] > threshold) | (n[idx] > threshold)).sum()
                total_clauses += 1
                if nl > 0:
                    non_empty += 1
                    total_literals += nl
                    ac += 1
            per_action[action_names[a]] = ac
        return {
            'total_clauses': total_clauses,
            'non_empty_clauses': non_empty,
            'avg_literals_per_clause': total_literals / max(non_empty, 1),
            'clauses_per_action': per_action,
        }


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAELogicConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    k: int = 10
    n_actions: int = 7
    initial_alpha: float = 1.0
    n_clauses_per_action: int = 10
    l0_penalty_weight: float = 1e-4

    beta_action: float = 5.0
    lambda_bimodal: float = 0.0
    bimodal_max: float = 0.3
    bimodal_warmup: int = 30  # epochs into Logic training
    bimodal_ramp: int = 80

    action_class_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # SAE pre-training components, retained for joint loss
    sae_lr: float = 1e-3
    lambda_sparsity: float = 5e-3
    alpha_recon: float = 1.0  

    # Logic training
    n_epochs: int = 400  # total epochs
    batch_size: int = 256
    logic_lr: float = 3e-3
    bottleneck_lr: float = 1e-3
    max_grad_norm: float = 5.0

    seed: int = 42
    use_ica_init: bool = True
    log_every: int = 10
    save_dir: str = "./sae_logic_joint_outputs"


# ============================================================================
# Model
# ============================================================================

class SAELogicAgentV3(nn.Module):
    def __init__(self, config: SAELogicConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device

        self.sae = OvercompleteSAE(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k,
        ).to(device)

        self.bottleneck = SigmoidBottleneck(
            n_features=config.hidden_dim,
            initial_alpha=config.initial_alpha,
        ).to(device)

        self.logic_layer = ProductTNormLogicLayer(
            n_features=config.hidden_dim,
            n_actions=config.n_actions,
            n_clauses_per_action=config.n_clauses_per_action,
            l0_penalty_weight=config.l0_penalty_weight,
        ).to(device)

        self.register_buffer('feature_mean', torch.zeros(config.input_dim))
        self.register_buffer('feature_std', torch.ones(config.input_dim))

        self.register_buffer('z_mean', torch.zeros(config.hidden_dim))
        self.register_buffer('z_std', torch.ones(config.hidden_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mean) / self.z_std

    def forward(self, x, normalize_input=False, return_features=False):
        if normalize_input:
            x = self.normalize_input(x)

        z_sparse, z_pre = self.sae.encode(x)
        z_normed = self.normalize_z(z_sparse)
        z_binary = self.bottleneck(z_normed)

        action_logits = self.logic_layer(z_binary)

        if return_features:
            x_recon = self.sae.decode(z_sparse)
            return action_logits, {
                'z_sparse': z_sparse,
                'z_pre': z_pre,
                'z_normed': z_normed,
                'z_binary': z_binary,
                'x_recon': x_recon,
            }
        return action_logits

    def extract_rules(self, concept_labels=None, action_names=None, threshold=0.3):
        return self.logic_layer.extract_rules(
            feature_names=concept_labels,
            action_names=action_names,
            threshold=threshold
        )


# ============================================================================
# Joint Training
# ============================================================================

def train_logic(
    model: SAELogicAgentV3,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: SAELogicConfig,
    device: str,
):
    """
    Train SAE, bottleneck, and logic layer jointly from scratch.
    """
    print("\n" + "=" * 70)
    print("JOINT TRAINING (SAE + Logic)")
    print("=" * 70)

    # Train all components
    optimizer = torch.optim.Adam([
        {'params': model.sae.parameters(), 'lr': config.sae_lr},
        {'params': model.bottleneck.parameters(), 'lr': config.bottleneck_lr},
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr},
    ])

    n_logic_epochs = config.n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_logic_epochs, eta_min=1e-5
    )

    class_weights = torch.tensor(
        config.action_class_weights, dtype=torch.float32, device=device
    )

    best_val_acc = 0.0
    best_model_state = None
    history = []
    patience = 0
    max_patience = 100

    for epoch_idx in range(n_logic_epochs):
        epoch = epoch_idx  # global epoch number

        model.sae.train()
        model.bottleneck.train()
        model.logic_layer.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)

            action_logits, features = model.forward(batch_x, return_features=True)

            # Action loss
            action_loss = F.cross_entropy(action_logits, batch_a, weight=class_weights)

            # SAE losses
            x_recon = features['x_recon']
            z_pre = features['z_pre']
            recon_loss = F.mse_loss(x_recon, batch_x)
            sparsity_loss = config.lambda_sparsity * z_pre.abs().mean()

            # Bimodality loss
            z_bin = features['z_binary']
            bimodal_raw = (z_bin * (1.0 - z_bin)).mean()
            if epoch_idx < config.bimodal_warmup:
                bimodal_weight = 0.0
            else:
                progress = min(1.0, (epoch_idx - config.bimodal_warmup) / max(config.bimodal_ramp, 1))
                bimodal_weight = config.bimodal_max * progress
            bimodal_loss = bimodal_weight * bimodal_raw

            # Logic complexity
            logic_complexity = model.logic_layer.complexity_penalty()

            total_loss = (
                config.alpha_recon * recon_loss +
                sparsity_loss +
                config.beta_action * action_loss +
                bimodal_loss +
                logic_complexity
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.sae.parameters()) + 
                list(model.bottleneck.parameters()) + 
                list(model.logic_layer.parameters()),
                config.max_grad_norm
            )
            optimizer.step()

            # Normalize decoder dictionary weights
            with torch.no_grad():
                model.sae._normalize_decoder()

            acc = (action_logits.argmax(1) == batch_a).float().mean()
            near_binary = ((z_bin < 0.05) | (z_bin > 0.95)).float().mean()

            train_info.append({
                'total_loss': total_loss.item(),
                'recon_loss': recon_loss.item(),
                'sparsity_loss': sparsity_loss.item(),
                'action_loss': action_loss.item(),
                'bimodal_loss': bimodal_loss.item(),
                'bimodal_weight': bimodal_weight,
                'bimodal_raw': bimodal_raw.item(),
                'logic_complexity': logic_complexity.item(),
                'accuracy': acc.item(),
                'near_binary_frac': near_binary.item(),
                'feature_density': (features['z_sparse'] > 0).float().mean().item(),
                'bottleneck_mean': z_bin.mean().item(),
                'bottleneck_std': z_bin.std().item(),
                'alpha_mean': model.bottleneck.get_sharpness().mean().item(),
            })

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)

        avg = avg_dict(train_info)
        avg['val_acc'] = val_acc
        avg['epoch'] = epoch
        history.append(avg)

        if (epoch + 1) % config.log_every == 0:
            lr_now = optimizer.param_groups[-1]['lr']
            print(
                f"  Epoch {epoch+1}/{config.n_epochs} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"Recon: {avg['recon_loss']:.4f} | "
                f"Act: {avg['action_loss']:.4f} | "
                f"TrainAcc: {avg['accuracy']:.3f} | "
                f"ValAcc: {val_acc:.3f} | "
                f"NearBin: {avg['near_binary_frac']:.3f} | "
                f"LR: {lr_now:.1e}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }
            patience = 0
        else:
            patience += 1

    return best_model_state, best_val_acc, history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            logits = model(batch_x)
            correct += (logits.argmax(1) == batch_a).sum().item()
            total += batch_a.size(0)
    return correct / total


def per_class_accuracy(model, loader, device, action_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(batch_a)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(f"\n  {'Action':<12} {'Count':>6} {'Correct':>8} {'Accuracy':>9}")
    print(f"  {'-'*38}")
    for i, name in enumerate(action_names):
        mask = all_labels == i
        count = mask.sum().item()
        if count > 0:
            correct = (all_preds[mask] == i).sum().item()
            print(f"  {name:<12} {count:>6} {correct:>8} {correct/count:>9.3f}")
        else:
            print(f"  {name:<12} {0:>6} {'N/A':>8} {'N/A':>9}")


def linear_probe(model, train_loader, val_loader, device, n_epochs=50, lr=1e-3):
    print(f"\n{'='*70}")
    print("LINEAR PROBE (information ceiling test)")
    print(f"{'='*70}")

    model.eval()
    n_features = model.config.hidden_dim
    n_actions = model.config.n_actions

    def collect(loader):
        zs, acts = [], []
        with torch.no_grad():
            for bx, ba in loader:
                bx = bx.to(device)
                z_sparse, _ = model.sae.encode(bx)
                z_normed = model.normalize_z(z_sparse)
                z_bin = model.bottleneck(z_normed)
                zs.append(z_bin.cpu())
                acts.append(ba)
        return torch.cat(zs, 0), torch.cat(acts, 0)

    train_z, train_a = collect(train_loader)
    val_z, val_a = collect(val_loader)

    print(f"  Features: {train_z.shape[1]}-d, "
          f"near-binary: {((train_z < 0.05) | (train_z > 0.95)).float().mean():.3f}")

    probe = nn.Linear(n_features, n_actions).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(train_z, train_a), batch_size=256, shuffle=True)

    for _ in range(n_epochs):
        probe.train()
        for bz, ba in dl:
            bz, ba = bz.to(device), ba.to(device)
            loss = F.cross_entropy(probe(bz), ba)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        tr_acc = (probe(train_z.to(device)).argmax(1).cpu() == train_a).float().mean().item()
        va_acc = (probe(val_z.to(device)).argmax(1).cpu() == val_a).float().mean().item()

    print(f"  Linear probe train acc: {tr_acc:.3f}")
    print(f"  Linear probe val acc:   {va_acc:.3f}")
    return tr_acc, va_acc


# ============================================================================
# Utilities
# ============================================================================

def avg_dict(dicts):
    keys = dicts[0].keys()
    return {k: np.mean([d[k] for d in dicts]) for k in keys}


def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SAE + Product T-Norm Logic Training (JOINT)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, [h['recon_loss'] for h in history], label='Recon', linewidth=2)
    ax.plot(epochs, [h['sparsity_loss'] for h in history], label='Sparsity', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('SAE Loss'); ax.set_title('SAE Metrics')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [h['accuracy'] for h in history], label='Train', linewidth=2)
    ax.plot(epochs, [h['val_acc'] for h in history], label='Val', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, [h['near_binary_frac'] for h in history], linewidth=2, color='teal')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Fraction near {0,1}')
    ax.set_title('Bottleneck Binarization'); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [h['bimodal_loss'] for h in history], label='Weighted', linewidth=2)
    ax.plot(epochs, [h['bimodal_raw'] for h in history], label='Raw', alpha=0.7)
    ax.plot(epochs, [h['bimodal_weight'] for h in history], label='Weight', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Bimodality')
    ax.set_title('Bimodality Loss & Weight'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [h['alpha_mean'] for h in history], linewidth=2, color='purple')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean α'); ax.set_title('Bottleneck Sharpness')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, [h['feature_density'] for h in history], label='SAE density', linewidth=2)
    ax.plot(epochs, [h['bottleneck_mean'] for h in history], label='Bottleneck mean', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Density'); ax.set_title('Feature Sparsity')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_joint.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load data ---
    print("\nLoading data...")
    data = torch.load(args.features_path, weights_only=False)
    features = data['features']
    actions = data['actions']
    print(f"  Raw features: {features.shape}, range=[{features.min():.2f}, {features.max():.2f}]")

    stage1_data = None
    if args.stage1_path and os.path.exists(args.stage1_path):
        print(f"  Loading Stage 1 from {args.stage1_path}...")
        stage1_data = torch.load(args.stage1_path, weights_only=False)
        feat_mean = stage1_data['feature_mean']
        feat_std = stage1_data['feature_std']
        features = (features - feat_mean) / feat_std
        print(f"  Normalized: range=[{features.min():.2f}, {features.max():.2f}]")
    else:
        feat_mean = features.mean(0)
        feat_std = features.std(0).clamp(min=1e-6)
        features = (features - feat_mean) / feat_std

    # --- Split ---
    n_train = int(0.9 * len(features))
    idx = torch.randperm(len(features), generator=torch.Generator().manual_seed(args.seed))
    train_ds = TensorDataset(features[idx[:n_train]], actions[idx[:n_train]])
    val_ds = TensorDataset(features[idx[n_train:]], actions[idx[n_train:]])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if getattr(args, 'save_training_data', False):
        training_data_path = os.path.join(args.save_dir, "training_data.pt")
        source_data = torch.load(args.features_path, weights_only=False)
        obs = source_data.get('observations', source_data.get('obs', None))

        save_dict = {
            'features': features,
            'actions': actions,
            'shuffle_indices': idx,
            'n_train': n_train,
            'feature_mean': feat_mean,
            'feature_std': feat_std,
            'pre_normalized': True,
        }
        if obs is not None:
            if isinstance(obs, torch.Tensor):
                save_dict['observations'] = obs
            else:
                save_dict['observations'] = torch.tensor(obs)
        torch.save(save_dict, training_data_path)

    action_names = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]

    # --- Config ---
    config = SAELogicConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        n_actions=len(action_names),
        n_clauses_per_action=args.n_clauses_per_action,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        use_ica_init=args.use_ica_init,
        bimodal_max=args.bimodal_max,
        bimodal_warmup=args.bimodal_warmup,
        bimodal_ramp=args.bimodal_ramp,
        l0_penalty_weight=args.l0_penalty,
        lambda_sparsity=args.lambda_sparsity,
        sae_lr=args.sae_lr,
        logic_lr=args.logic_lr,
        bottleneck_lr=args.bottleneck_lr,
        action_class_weights=tuple(args.action_class_weights),
        max_grad_norm=args.max_grad_norm,
        beta_action=args.beta_action,
    )

    # --- Model ---
    print("\nInitializing model...")
    model = SAELogicAgentV3(config, device=device)
    model.set_normalization(feat_mean, feat_std)

    if stage1_data is not None and config.use_ica_init:
        print("Initializing SAE from Stage 1...")
        sae_config = SAEConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            k=config.k,
            n_actions=config.n_actions,
            use_ica_init=config.use_ica_init,
        )
        init_from_stage1(model.sae, stage1_data, sae_config)

    print(f"\nArchitecture: {config.input_dim} → SAE({config.hidden_dim}, k={config.k}) → "
          f"FixedNorm → Sigmoid → Logic({config.n_clauses_per_action} clauses/action) → {config.n_actions}")
    print(f"  Joint train: {config.n_epochs} epochs")

    model.to(device)

    # ============================
    # JOINT TRAINING
    # ============================
    best_state, best_acc, history = train_logic(
        model, train_loader, val_loader, config, device
    )

    print(f"\n{'='*70}")
    print(f"Training complete! Best val accuracy: {best_acc:.3f}")
    print(f"{'='*70}")

    if best_state is not None:
        model.load_state_dict(best_state['model'])
    else:
        best_state = {'model': model.state_dict(), 'epoch': config.n_epochs-1, 'val_acc': best_acc}

    # --- Analysis ---
    per_class_accuracy(model, val_loader, device, action_names)
    linear_probe(model, train_loader, val_loader, device)

    # --- Rules ---
    rules = model.extract_rules(action_names=action_names, threshold=args.threshold)
    
    # --- Save ---
    save_path = os.path.join(args.save_dir, "sae_logic_joint_model.pt")
    torch.save({
        'model_state': best_state['model'],
        'config': asdict(config),
        'rules': rules,
        'best_val_acc': best_acc,
        'feature_mean': feat_mean,
        'feature_std': feat_std,
        'z_mean': model.z_mean.cpu(),
        'z_std': model.z_std.cpu(),
    }, save_path)

    with open(os.path.join(args.save_dir, "learned_rules.json"), 'w') as f:
        json.dump(rules, f, indent=2)

    plot_training_history(history, args.save_dir)

    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE + Product T-Norm Logic JOINT")

    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--stage1_path", type=str, default=None)

    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n_clauses_per_action", type=int, default=10)

    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sae_lr", type=float, default=1e-3)
    parser.add_argument("--logic_lr", type=float, default=3e-3)
    parser.add_argument("--bottleneck_lr", type=float, default=1e-3)

    parser.add_argument("--beta_action", type=float, default=5.0)
    parser.add_argument("--bimodal_max", type=float, default=0.3)
    parser.add_argument("--bimodal_warmup", type=int, default=30)
    parser.add_argument("--bimodal_ramp", type=int, default=80)

    parser.add_argument("--l0_penalty", type=float, default=1e-4)
    parser.add_argument("--lambda_sparsity", type=float, default=5e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    
    # [MỚI] Nhận tham số Temperature từ Terminal
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_end", type=float, default=0.1)
    parser.add_argument("--tau_anneal_epochs", type=int, default=200)

    parser.add_argument("--save_dir", type=str, default="./sae_logic_joint_outputs")
    parser.add_argument(
        "--action_class_weights", type=float, nargs=7,
        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    parser.add_argument("--save_training_data", action="store_true",
                    help="Save normalized features + observations for visualization")
    parser.add_argument("--use_ica_init", action="store_true", default=True,
                    help="Use ICA initialization for SAE (default: True)")
    parser.add_argument("--no_ica_init", action="store_false", dest="use_ica_init",
                    help="Disable ICA initialization")
    parser.add_argument("--threshold", type=float, default=0.3,
                    help="Threshold for rule extraction")
    args = parser.parse_args()
    main(args)