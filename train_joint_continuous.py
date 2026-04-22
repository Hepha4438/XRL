"""
SAE + Continuous T-Norm Neural Logic Training (JOINT TRAINING BASELINE)
=============================================================
Adapted for Continuous Action Spaces (Additive Rule Regression).
Replaces CrossEntropy with MSE and argmax logic with Fuzzy Logic.

Usage:
    python train_joint_continuous.py \
        --features_path ./collected_data/continuous_data.pt \
        --hidden_dim 300 --k 50 \
        --n_clauses 20 \
        --n_epochs 400 \
        --save_dir ./sae_logic_continuous_outputs \
        --entropy_weight 0.05
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
# Continuous Product T-Norm Logic Layer
# ============================================================================
class ContinuousTNormLogicLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        action_dim: int, 
        n_clauses: int = 20, 
        l0_penalty_weight: float = 1e-4,
    ):
        super().__init__()
        self.n_features = n_features
        self.action_dim = action_dim
        self.n_clauses = n_clauses
        self.l0_penalty_weight = l0_penalty_weight

        # 1. IF CONDITIONS
        self.w_pos = nn.Parameter(torch.randn(n_clauses, n_features) * 0.01 - 3.0)
        self.w_neg = nn.Parameter(torch.randn(n_clauses, n_features) * 0.01 - 3.0)
        self.clause_weight = nn.Parameter(torch.ones(n_clauses) * 2.0)

        # 2. THEN VALUES (Additive Rules)
        self.action_values = nn.Parameter(torch.randn(n_clauses, action_dim) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(action_dim))

    def _get_selection_probs(self):
        absent_logit = torch.zeros_like(self.w_pos)
        logits = torch.stack([self.w_pos, self.w_neg, absent_logit], dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs[..., 0], probs[..., 1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        p, n = self._get_selection_probs()

        f = features.unsqueeze(1)
        p_ex = p.unsqueeze(0)
        n_ex = n.unsqueeze(0)

        literals = p_ex * f + n_ex * (1.0 - f) + (1.0 - p_ex - n_ex)
        log_literals = torch.log(literals + 1e-8)
        log_clause_sum = log_literals.sum(dim=-1)

        # Activation strength [0.0, 1.0]
        clauses_activation = torch.sigmoid(log_clause_sum + self.clause_weight.unsqueeze(0)) 

        # Additive Rule Regression: Action = Bias + Sum(Activation_i * Action_Value_i)
        continuous_action = self.base_bias + torch.matmul(clauses_activation, self.action_values)
        
        # Bound actions between [-1, 1]
        return torch.tanh(continuous_action)

    def complexity_penalty(self) -> torch.Tensor:
        p, n = self._get_selection_probs()
        return self.l0_penalty_weight * (p + n).mean()

    def entropy_penalty(self) -> torch.Tensor:
        p, n = self._get_selection_probs()
        a = 1.0 - p - n 
        entropy = - (p * torch.log(p + 1e-8) + n * torch.log(n + 1e-8) + a * torch.log(a + 1e-8))
        return entropy.sum()

    def extract_rules(self, feature_names=None, threshold=0.3):
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(self.n_features)]

        p, n = self._get_selection_probs()
        p, n = p.detach().cpu().numpy(), n.detach().cpu().numpy()
        acts = self.action_values.detach().cpu().numpy()
        bias = self.base_bias.detach().cpu().numpy()

        rules = {
            "Base_Bias": [float(b) for b in bias],
            "Active_Rules": []
        }
        
        for c in range(self.n_clauses):
            lits = []
            for i in range(self.n_features):
                if p[c, i] > threshold:
                    lits.append(f"{feature_names[i]}")
                elif n[c, i] > threshold:
                    lits.append(f"¬{feature_names[i]}")
            
            if lits:
                rule_str = f"IF ({' ∧ '.join(lits)}) THEN Add {[round(float(v), 3) for v in acts[c]]}"
                rules["Active_Rules"].append(rule_str)
                
        if not rules["Active_Rules"]:
            rules["Active_Rules"].append("(No rules active)")
            
        return rules


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class SAELogicConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    k: int = 10
    action_dim: int = 2  # Set based on env (e.g., LunarLander is 2)
    initial_alpha: float = 1.0
    n_clauses: int = 20
    l0_penalty_weight: float = 1e-4

    beta_action: float = 5.0
    lambda_bimodal: float = 0.0
    bimodal_max: float = 0.3
    bimodal_warmup: int = 30
    bimodal_ramp: int = 80
    entropy_weight: float = 0.005

    sae_lr: float = 1e-3
    lambda_sparsity: float = 5e-3
    alpha_recon: float = 1.0  

    n_epochs: int = 400
    batch_size: int = 256
    logic_lr: float = 3e-3
    bottleneck_lr: float = 1e-3
    max_grad_norm: float = 5.0

    seed: int = 42
    use_ica_init: bool = True
    log_every: int = 10
    save_dir: str = "./sae_logic_continuous_outputs"


# ============================================================================
# Model
# ============================================================================
class SAELogicAgentContinuous(nn.Module):
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

        self.logic_layer = ContinuousTNormLogicLayer(
            n_features=config.hidden_dim,
            action_dim=config.action_dim,
            n_clauses=config.n_clauses,
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

        predicted_actions = self.logic_layer(z_binary)

        if return_features:
            x_recon = self.sae.decode(z_sparse)
            return predicted_actions, {
                'z_sparse': z_sparse,
                'z_pre': z_pre,
                'z_normed': z_normed,
                'z_binary': z_binary,
                'x_recon': x_recon,
            }
        return predicted_actions


# ============================================================================
# Joint Training for Continuous Control
# ============================================================================
def train_logic_continuous(model, train_loader, val_loader, config, device):
    print("\n" + "=" * 70)
    print("JOINT TRAINING (CONTINUOUS ACTIONS) WITH MSE + ENTROPY PENALTY")
    print("=" * 70)

    optimizer = torch.optim.Adam([
        {'params': model.sae.parameters(), 'lr': config.sae_lr},
        {'params': model.bottleneck.parameters(), 'lr': config.bottleneck_lr},
        {'params': model.logic_layer.parameters(), 'lr': config.logic_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=1e-5)

    best_val_mse = float('inf')
    best_model_state = None
    history = []

    for epoch_idx in range(config.n_epochs):
        model.train()
        train_info = []

        for batch_x, batch_a in train_loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            predicted_actions, features = model.forward(batch_x, return_features=True)

            # Continuous regression loss
            action_loss = F.mse_loss(predicted_actions, batch_a)
            
            x_recon = features['x_recon']
            z_pre = features['z_pre']
            recon_loss = F.mse_loss(x_recon, batch_x)
            sparsity_loss = config.lambda_sparsity * z_pre.abs().mean()

            z_bin = features['z_binary']
            bimodal_raw = (z_bin * (1.0 - z_bin)).mean()
            
            if epoch_idx < config.bimodal_warmup:
                bimodal_weight = current_entropy_weight = 0.0
            else:
                progress = min(1.0, (epoch_idx - config.bimodal_warmup) / max(config.bimodal_ramp, 1))
                bimodal_weight = config.bimodal_max * progress
                current_entropy_weight = config.entropy_weight * progress
                
            bimodal_loss = bimodal_weight * bimodal_raw
            logic_complexity = model.logic_layer.complexity_penalty()
            logic_entropy = model.logic_layer.entropy_penalty()

            total_loss = (config.alpha_recon * recon_loss + sparsity_loss + config.beta_action * action_loss + 
                          bimodal_loss + logic_complexity + (current_entropy_weight * logic_entropy))

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad(): model.sae._normalize_decoder()

            near_binary = ((z_bin < 0.05) | (z_bin > 0.95)).float().mean()

            train_info.append({
                'total_loss': total_loss.item(), 'recon_loss': recon_loss.item(), 'sparsity_loss': sparsity_loss.item(),
                'action_mse': action_loss.item(), 'bimodal_loss': bimodal_loss.item(), 'logic_entropy': logic_entropy.item(),
                'current_entropy_weight': current_entropy_weight, 'near_binary_frac': near_binary.item(),
                'feature_density': (features['z_sparse'] > 0).float().mean().item(), 'bottleneck_mean': z_bin.mean().item(),
                'alpha_mean': model.bottleneck.get_sharpness().mean().item()
            })

        scheduler.step()
        val_mse = evaluate_mse(model, val_loader, device)
        avg = {k: np.mean([d[k] for d in train_info]) for k in train_info[0].keys()}
        avg.update({'val_mse': val_mse, 'epoch': epoch_idx})
        history.append(avg)

        # Save model if validation MSE improves (lower is better)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_state = {'model': model.state_dict(), 'epoch': epoch_idx, 'val_mse': val_mse}
            if (epoch_idx + 1) % config.log_every == 0:
                 print(f"    [Best Model Updated] Epoch {epoch_idx+1}: Val MSE {val_mse:.4f}")

        if (epoch_idx + 1) % config.log_every == 0:
            lr_now = optimizer.param_groups[-1]['lr']
            print(
                f"  Epoch {epoch_idx+1}/{config.n_epochs} | "
                f"Loss: {avg['total_loss']:.4f} | "
                f"Recon: {avg['recon_loss']:.4f} | "
                f"TrainMSE: {avg['action_mse']:.4f} | "
                f"ValMSE: {val_mse:.4f} | "
                f"NearBin: {avg['near_binary_frac']:.3f} | "
                f"Ent: {avg['logic_entropy']:.3f} | "
                f"LR: {lr_now:.1e}"
            )

    return best_model_state, best_val_mse, history

def evaluate_mse(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_a in loader:
            batch_x, batch_a = batch_x.to(device), batch_a.to(device)
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_a, reduction='sum')
            total_loss += loss.item()
            count += batch_a.size(0) * batch_a.size(1)
    return total_loss / count

def linear_regression_probe(model, train_loader, val_loader, device, n_epochs=50, lr=1e-3):
    print(f"\n{'='*70}")
    print("LINEAR REGRESSION PROBE (information ceiling test)")
    print(f"{'='*70}")

    model.eval()
    n_features = model.config.hidden_dim
    action_dim = model.config.action_dim

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

    probe = nn.Linear(n_features, action_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(train_z, train_a), batch_size=256, shuffle=True)

    for _ in range(n_epochs):
        probe.train()
        for bz, ba in dl:
            bz, ba = bz.to(device), ba.to(device)
            loss = F.mse_loss(probe(bz), ba)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        tr_mse = F.mse_loss(probe(train_z.to(device)), train_a.to(device)).item()
        va_mse = F.mse_loss(probe(val_z.to(device)), val_a.to(device)).item()

    print(f"  Linear probe train MSE: {tr_mse:.4f}")
    print(f"  Linear probe val MSE:   {va_mse:.4f}")
    return tr_mse, va_mse

def plot_training_history_continuous(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SAE + Continuous Logic Training (JOINT)", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, [h['recon_loss'] for h in history], label='Recon', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('SAE Loss'); axes[0, 0].set_title('SAE Metrics')

    axes[0, 1].plot(epochs, [h['action_mse'] for h in history], label='Train MSE', linewidth=2)
    axes[0, 1].plot(epochs, [h['val_mse'] for h in history], label='Val MSE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MSE Loss'); axes[0, 1].set_title('Action Regression Error')
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, [h['near_binary_frac'] for h in history], linewidth=2, color='teal')
    axes[0, 2].set_title('Bottleneck Binarization')

    axes[1, 0].plot(epochs, [h['logic_entropy'] for h in history], label='Entropy Penalty', color='orange')
    axes[1, 0].set_title('Entropy Optimization')

    axes[1, 1].plot(epochs, [h['alpha_mean'] for h in history], linewidth=2, color='purple')
    axes[1, 1].set_title('Bottleneck Sharpness')

    axes[1, 2].plot(epochs, [h['feature_density'] for h in history], label='SAE density')
    axes[1, 2].set_title('Feature Sparsity')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_continuous.png"))
    plt.close()

# ============================================================================
# Main
# ============================================================================
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    print("\nLoading continuous data...")
    data = torch.load(args.features_path, weights_only=False)
    features = data['features']
    actions = data['actions']  # Should be shape [N, action_dim]
    
    action_dim = actions.shape[1]
    print(f"  Feature dim: {features.shape}, Action dim: {action_dim}")

    feat_mean = features.mean(0)
    feat_std = features.std(0).clamp(min=1e-6)
    features = (features - feat_mean) / feat_std

    n_train = int(0.9 * len(features))
    idx = torch.randperm(len(features))
    train_ds = TensorDataset(features[idx[:n_train]], actions[idx[:n_train]])
    val_ds = TensorDataset(features[idx[n_train:]], actions[idx[n_train:]])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    config = SAELogicConfig(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        k=args.k,
        action_dim=action_dim,
        n_clauses=args.n_clauses,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        entropy_weight=args.entropy_weight,
    )

    model = SAELogicAgentContinuous(config, device=device)
    model.set_normalization(feat_mean, feat_std)
    model.to(device)

    best_state, best_mse, history = train_logic_continuous(
        model, train_loader, val_loader, config, device
    )

    print(f"\nTraining complete! Best val MSE: {best_mse:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state['model'])

    linear_regression_probe(model, train_loader, val_loader, device)
    rules = model.extract_rules(threshold=args.threshold)
    
    save_path = os.path.join(args.save_dir, "sae_logic_continuous_model.pt")
    torch.save({
        'model_state': best_state['model'],
        'config': asdict(config),
        'rules': rules,
        'best_val_mse': best_mse,
        'feature_mean': feat_mean,
        'feature_std': feat_std,
    }, save_path)

    with open(os.path.join(args.save_dir, "continuous_learned_rules.json"), 'w') as f:
        json.dump(rules, f, indent=2)

    plot_training_history_continuous(history, args.save_dir)
    print(f"Saved model and rules to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAE + Continuous Logic JOINT")
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--n_clauses", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy_weight", type=float, default=0.005)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--save_dir", type=str, default="./sae_logic_continuous_outputs")
    
    args = parser.parse_args()
    main(args)