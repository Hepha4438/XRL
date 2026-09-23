#!/usr/bin/env python3
"""
verify_fidelity.py -- Empirical validation of the Soft--Hard Fidelity theorem
(Theorem 1 in LUCID.tex, Table 5 "Empirical Fidelity Validation").

What is computed (exactly as stated in the paper, Sec. "Theoretical Analysis"):

  Soft surrogate
      c_j^soft(x) = sigma(alpha_j (z_j(x) - beta_j))
      l^soft_{a,i,j} = p c + n (1-c) + u,             u = 1 - p - n
      L_{a,i}(x)     = sum_j log(l^soft_{a,i,j} + eps),  eps = 1e-8
      C^soft_{a,i}   = sigma(b~_{a,i} + L_{a,i}),       b~ = min(b, b_max=5)
      s^soft_a       = sum_i C^soft_{a,i},              pi^soft = argmax_a s^soft_a

  Hard rules (two DIFFERENT thresholds, both 0.66 by default)
      tau   = --threshold      (selector threshold): P = {j : p > tau},  N = {j : n > tau}
      theta = --hard_threshold (concept threshold) : c_j^hard = 1[c_j^soft > theta]
      (i)  contradictory clause (P & N != {}) -> False
      (ii) universal clause (P | N == {})      -> True (constant prior sigma(b~))
      (iii) clipped bias b~ in both soft and hard scores
      C^hard_{a,i} = sigma(b~) * 1[T_{a,i}(x)],  s^hard_a = sum_i C^hard_{a,i}
      pi^hard = argmax_a s^hard_a on X_trig, a_fallback otherwise

  Per-clause bound Delta_{a,i}(x)
      true clause : |sigma(b~) - sigma(b~ + L)|
      false clause: sigma(b~ + log(l^soft_{j*} + eps) + (D-1) log(1+eps)),
                    j* = argmin_{j in V_{a,i}(x)} l^soft_{a,i,j}
                    V  = violated literals (P with c_hard=0, N with c_hard=1)

  Certificate
      pairwise (theorem):  Delta_{a*} + Delta_a < s^hard_{a*} - s^hard_a   for all a != a*
      simple  (sufficient): Delta_total = Delta_{a*} + max_{a!=a*} Delta_a < gamma(x)
      X_cert = {x in X_trig : gamma(x) > 0, pairwise holds}
      Pr[pi^soft != pi^hard] <= 1 - Pr[X_cert]
      Triangle: 1 - AF_hard <= (1 - AF_soft) + Pr[soft != hard] <= (1 - AF_soft) + (1 - Pr[X_cert])

Built-in self-checks (should all be 0 / tiny; if not, either the code or the
theorem is wrong and the numbers must NOT go into the paper):
  * max |s^soft (re-implementation) - s^soft (model.forward)|
  * #(state, clause) pairs with |C^soft - C^hard| > Delta_{a,i}        (Lemma 1)
  * #(state, action) pairs with |s^soft_a - s^hard_a| > Delta_a        (Lemma 2)
  * #certified states with pi^soft != pi^hard                           (Theorem 1)
  * #(clean state, clause) pairs with Delta_{a,i} > Dbar_{a,i}           (Corollary)

  Cleanliness certificate (Corollary; uses only the hard truth pattern + cleanliness)
      x is eps_bar-clean if every rule concept has min(c_j, 1-c_j) < eps_bar <= min(theta, 1-theta).
      On a clean state:  true clause  Delta <= sigma(b~)(1 - exp(-(A + U)))   [U: ignored-literal drag]
                         false clause Delta <= sigma(b~ + log(1 - s(1-eps_bar) + eps) + (D-1)log(1+eps))
      state-wise: s = strongest violated literal at x;  global: s = weakest literal of the clause,
      Dbar^G_a = sum_i max(true, false) (data-independent).
      Pr[soft != hard] <= Pr[not clean] + Pr[clean, not certified by Dbar].

  Invariance over --invariance_range [0.5, 0.9]
      #selectors in (lo, hi]  == 0  -> identical rule set for every tau in [lo, hi]
      #states with a rule-concept activation in (lo, hi] == 0 -> identical hard policy for every theta in [lo, hi]

Supported environments (IDs used by the teacher/feature collection in this repo):
  MiniGrid-DoorKey-6x6-v0, MiniGrid-Dynamic-Obstacles-5x5-v0, PixelCartPole-v0,
  PongNoFrameskip-v4, BoxingNoFrameskip-v4

Usage (both thresholds default to 0.66):
  python experiments/theorem/verify_fidelity.py \
      --model_path ./outputs/lucid_model_pong/sae_logic_joint_model.pt \
      --data_path ./heldout_pong/collected_data.pt \
      --env_name PongNoFrameskip-v4 \
      --threshold 0.66 --hard_threshold 0.66 \
      --output_dir ./experiments/theorem/results_pong

Outputs (in --output_dir):
  theorem2_analytics.npz   per-state arrays + histograms + theta-sweep + diagnostics
  theorem2_summary.json    all scalar results
  table5_row.tex           one LaTeX row for Table 5 (clean states, ties, actual disagreement, 3 bounds)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Constants that must match train_joint.py (ProductTNormLogicLayer.forward).
# The model-forward consistency check below fails loudly if they drift.
EPS = 1e-8
B_MAX = 5.0

CANONICAL_ENVS = {
    "MiniGrid-DoorKey-6x6-v0": "DoorKey-6x6",
    "MiniGrid-Dynamic-Obstacles-5x5-v0": "Dynamic-Obs-5x5",
    "PixelCartPole-v0": "PixelCartPole",
    "PongNoFrameskip-v4": "Pong",
    "BoxingNoFrameskip-v4": "Boxing",
}

ACTION_NAMES = {
    "MiniGrid": ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"],
    "CartPole": ["Left", "Right"],
    "Pong": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
    "Boxing": ["NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
               "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
               "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"],
}


# ============================================================================
# Environment bookkeeping (labels only -- this script never creates an env)
# ============================================================================

def check_env_name(env_name):
    if not env_name:
        return "", "unknown"
    if env_name in CANONICAL_ENVS:
        return env_name, CANONICAL_ENVS[env_name]
    for canon, short in CANONICAL_ENVS.items():
        key = short.split("-")[0].lower()
        if key in env_name.lower():
            print(f"[WARN] env_name '{env_name}' is not the ID used by the teacher / feature "
                  f"collection in this repo; using '{canon}'.")
            if "NoFrameskip-v0" in env_name:
                print("       Note: *NoFrameskip-v0 uses sticky actions (repeat_action_probability=0.25); "
                      "the PPO teachers were trained on *NoFrameskip-v4 (no sticky actions).")
            return canon, short
    print(f"[WARN] Unknown env_name '{env_name}', used as a label only.")
    return env_name, env_name


def get_action_names(env_name, n_actions):
    for key, names in ACTION_NAMES.items():
        if key in env_name and len(names) == n_actions:
            return names
    return [f"a{i}" for i in range(n_actions)]


# ============================================================================
# Loading
# ============================================================================

def pick_device(requested, dtype):
    if requested != "auto":
        dev = requested
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "mps" and dtype == torch.float64:
        print("[WARN] MPS does not support float64; falling back to CPU.")
        dev = "cpu"
    return dev


def load_lucid(model_path, device, dtype):
    from train_joint import SAELogicAgentV3, SAELogicConfig
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = SAELogicConfig(**ckpt["config"])
    model = SAELogicAgentV3(config, device="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.set_normalization(ckpt["feature_mean"].float(), ckpt["feature_std"].float())
    if "z_mean" in ckpt:
        model.z_mean.copy_(ckpt["z_mean"])
        model.z_std.copy_(ckpt["z_std"])
    model = model.to(device=device, dtype=dtype).eval()
    return model, config


def load_data(data_path, split):
    try:
        data = torch.load(data_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        data = torch.load(data_path, map_location="cpu", weights_only=False)
    features = data["features"]
    actions = data.get("actions", None)
    pre_normalized = bool(data.get("pre_normalized", False))

    if split != "all":
        if "shuffle_indices" not in data or "n_train" not in data:
            raise ValueError("--split train/val requires a training_data.pt saved by "
                             "train_joint.py --save_training_data")
        idx = data["shuffle_indices"]
        n_train = int(data["n_train"])
        idx = idx[n_train:] if split == "val" else idx[:n_train]
        features = features[idx]
        if actions is not None:
            actions = actions[idx]

    features = torch.as_tensor(features).float().contiguous()
    if actions is not None:
        actions = torch.as_tensor(actions).long().view(-1)
        if len(actions) != len(features):
            print(f"[WARN] #actions ({len(actions)}) != #features ({len(features)}); ignoring teacher actions.")
            actions = None
    return features, actions, pre_normalized


# ============================================================================
# Rule structure at selector threshold tau
# ============================================================================

@torch.no_grad()
def extract_structure(model, tau):
    ll = model.logic_layer
    p, n = ll._get_selection_probs()            # identical to the forward pass
    u = 1.0 - p - n                               # as in the forward pass
    b_tilde = torch.clamp(ll.clause_weight, max=B_MAX)
    P = p > tau
    N = n > tau
    active = P | N
    return {
        "p": p, "n": n, "u": u, "b_tilde": b_tilde,
        "P": P, "N": N, "active": active,
        "contradictory": (P & N).any(-1),
        "universal": ~active.any(-1),
        "rule_concepts": active.any(0),        # concepts used by any clause
        "n_actions": ll.n_actions,
        "n_clauses": ll.n_clauses_per_action,
        "D": p.shape[1],
        "clause_action": torch.arange(p.shape[0], device=p.device) // ll.n_clauses_per_action,
    }


def rule_stats(S):
    active = S["active"]
    contra = S["contradictory"]
    univ = S["universal"]
    real = ~contra & ~univ
    return {
        "n_clauses_total": int(active.shape[0]),
        "n_nonuniversal_clauses": int(real.sum()),
        "n_universal_clauses": int(univ.sum()),
        "n_contradictory_clauses": int(contra.sum()),
        "n_literals": int(active[real].sum()),
        "n_concepts": int(active[real].any(0).sum()),
    }


# ============================================================================
# Core computation (pure tensor math; unit-tested separately)
# ============================================================================

def soft_part(c_soft, S, eps=EPS):
    """Soft clause/score quantities. c_soft: [B, D]."""
    B = c_soft.shape[0]
    A, K = S["n_actions"], S["n_clauses"]
    c = c_soft.unsqueeze(1)                                    # [B,1,D]
    l_soft = S["p"] * c + S["n"] * (1.0 - c) + S["u"]          # [B,C,D]
    log_l = torch.log(l_soft + eps)                            # [B,C,D]
    L = log_l.sum(-1)                                          # [B,C]
    C_soft = torch.sigmoid(S["b_tilde"] + L)                   # [B,C]
    s_soft = C_soft.reshape(B, A, K).sum(-1)                   # [B,A]
    return {"l_soft": l_soft, "log_l": log_l, "L": L, "C_soft": C_soft, "s_soft": s_soft}


def hard_part(c_soft, soft, S, theta, fallback, eps=EPS):
    """Hard rules, per-clause bounds and certificate at concept threshold theta."""
    B, D = c_soft.shape
    A, K = S["n_actions"], S["n_clauses"]
    dtype = c_soft.dtype
    inf = torch.tensor(float("inf"), dtype=dtype, device=c_soft.device)

    l_soft, L, C_soft, s_soft = soft["l_soft"], soft["L"], soft["C_soft"], soft["s_soft"]
    sig_b = torch.sigmoid(S["b_tilde"])                        # [C]

    # ---- hard clauses -------------------------------------------------------
    c_hard = (c_soft > theta).unsqueeze(1)                     # [B,1,D]
    viol = (S["P"] & ~c_hard) | (S["N"] & c_hard)              # [B,C,D]  V_{a,i}(x)
    T = ~viol.any(-1)                                          # [B,C]  (contradictory -> False, universal -> True)
    C_hard = sig_b * T.to(dtype)                               # [B,C]
    s_hard = C_hard.reshape(B, A, K).sum(-1)                   # [B,A]

    # ---- per-clause bound Delta_{a,i} --------------------------------------
    delta_true = (sig_b - C_soft).abs()
    l_viol = torch.where(viol, l_soft, inf)
    l_star, j_star = l_viol.min(-1)                            # [B,C]
    l_star = torch.where(T, torch.ones_like(l_star), l_star)   # dummy for true clauses (not used)
    delta_false = torch.sigmoid(S["b_tilde"] + torch.log(l_star + eps) + (D - 1) * math.log1p(eps))
    delta = torch.where(T, delta_true, delta_false)            # [B,C]
    delta_a = delta.reshape(B, A, K).sum(-1)                   # [B,A]

    # ---- hard policy, margin ------------------------------------------------
    trig = T.any(-1)                                           # X_trig
    a_star = s_hard.argmax(-1)                                 # first max (deterministic tie-break)
    top2 = torch.topk(s_hard, k=min(2, A), dim=-1).values
    gamma = top2[:, 0] - top2[:, 1] if A > 1 else top2[:, 0]
    pi_hard = torch.where(trig, a_star, torch.full_like(a_star, fallback))
    pi_soft = s_soft.argmax(-1)

    # ---- certificates -------------------------------------------------------
    idx = a_star.unsqueeze(-1)
    s_star = s_hard.gather(-1, idx)
    d_star = delta_a.gather(-1, idx)                           # [B,1]
    slack = (s_star - s_hard) - (d_star + delta_a)             # [B,A]
    slack = slack.scatter(-1, idx, inf.expand_as(idx).clone())
    pair_slack = slack.min(-1).values                          # >0 <=> pairwise condition
    other = delta_a.scatter(-1, idx, (-inf).expand_as(idx).clone())
    d_other, a_other = other.max(-1)
    delta_total = d_star.squeeze(-1) + d_other

    valid = trig & (gamma > 0)
    cert = valid & (pair_slack > 0)
    cert_simple = valid & (delta_total < gamma)

    # ---- Lemma checks -------------------------------------------------------
    lemma1_excess = (C_soft - C_hard).abs() - delta            # must be <= 0
    lemma2_excess = (s_soft - s_hard).abs() - delta_a          # must be <= 0

    return {
        "c_hard": c_hard.squeeze(1), "viol": viol, "T": T, "s_hard": s_hard,
        "delta": delta, "delta_a": delta_a, "j_star": j_star, "l_star": l_star,
        "trig": trig, "a_star": a_star, "a_other": a_other, "gamma": gamma,
        "pi_hard": pi_hard, "pi_soft": pi_soft,
        "pair_slack": pair_slack, "delta_star": d_star.squeeze(-1), "delta_other": d_other,
        "delta_total": delta_total, "cert": cert, "cert_simple": cert_simple,
        "tie": trig & (gamma <= 0),
        "lemma1_excess": lemma1_excess, "lemma2_excess": lemma2_excess,
    }


def sweep_counts(H, teacher):
    """Scalar counts for the theta sweep."""
    agree = H["pi_soft"] == H["pi_hard"]
    out = {
        "n": int(agree.numel()),
        "cert": int(H["cert"].sum()), "cert_simple": int(H["cert_simple"].sum()),
        "agree": int(agree.sum()), "trig": int(H["trig"].sum()), "tie": int(H["tie"].sum()),
        "violations": int((H["cert"] & ~agree).sum()),
        "hard_teacher": 0,
    }
    if teacher is not None:
        out["hard_teacher"] = int((H["pi_hard"] == teacher).sum())
    return out


# ============================================================================
# Cleanliness certificate (Corollary): bounds that do NOT use the soft forward pass
# ============================================================================
#
# A state x is eps_bar-clean if every rule concept j (j in some P or N) satisfies
# min(c_j, 1 - c_j) < eps_bar, with eps_bar <= min(theta, 1 - theta). On such a state:
#   ignored literal  (j not in P u N):  l >= u
#   satisfied j in P (c >= 1-eps_bar):  l >= 1 - n - p*eps_bar
#   satisfied j in N (c <= eps_bar)  :  l >= 1 - p - n*eps_bar
#   violated  j in P (c <= eps_bar)  :  l <= 1 - p*(1 - eps_bar)
#   violated  j in N (c >= 1-eps_bar):  l <= 1 - n*(1 - eps_bar)
# hence, with U = sum_{ignored} -log(u+eps) and A = sum_{active} -log(lower+eps),
#   true clause : Delta <= Dbar_true  = max( sigma(b~)(1 - e^{-(A+U)}),  sigma(b~ + D log(1+eps)) - sigma(b~) )
#   false clause: Delta <= Dbar_false = sigma(b~ + log(1 - s(1-eps_bar) + eps) + (D-1) log(1+eps)),
#                 s = selector strength of a violated literal (state-wise: the strongest one;
#                 global: the weakest literal of the clause).
# State-wise corollary: uses only the hard truth pattern of x (from c_hard) + cleanliness.
# Global corollary    : Dbar^G_a = sum_i max(Dbar_true, Dbar_false^G) is data-independent;
#                       any clean, non-tied state with hard margins exceeding Dbar^G is certified.
#   Pr[pi_soft != pi_hard] <= Pr[x not clean] + Pr[x clean, not certified].

def pairwise_slack(s_hard, a_star, delta_a):
    """min_{a != a*} (s_a* - s_a) - (Delta_a* + Delta_a); delta_a: [B,A] (or broadcastable)."""
    inf = torch.tensor(float("inf"), dtype=s_hard.dtype, device=s_hard.device)
    idx = a_star.unsqueeze(-1)
    delta_a = delta_a + torch.zeros_like(s_hard)                    # broadcast [A] -> [B,A]
    slack = (s_hard.gather(-1, idx) - s_hard) - (delta_a.gather(-1, idx) + delta_a)
    slack = slack.scatter(-1, idx, inf.expand_as(idx).clone())
    return slack.min(-1).values


def structural_constants(S, eps_bar, eps=EPS):
    """Per-clause, data-independent constants of the Cleanliness corollary."""
    p, n, u, bt = S["p"], S["n"], S["u"], S["b_tilde"]
    P, N, act = S["P"], S["N"], S["active"]
    D, A, K = S["D"], S["n_actions"], S["n_clauses"]
    log1pe = math.log1p(eps)
    sig_b = torch.sigmoid(bt)

    U = (-torch.log(u + eps) * ~act).sum(-1)                                   # [C] structural drag
    lo_pos = 1.0 - n - p * eps_bar
    lo_neg = 1.0 - p - n * eps_bar
    Aact = (-torch.log(lo_pos + eps) * P).sum(-1) + (-torch.log(lo_neg + eps) * N).sum(-1)   # [C]
    d_true = torch.maximum(sig_b * (1.0 - torch.exp(-(Aact + U))),
                           torch.sigmoid(bt + D * log1pe) - sig_b)

    big = torch.ones_like(p) * 2.0
    s_min = torch.minimum(torch.where(P, p, big).min(-1).values,
                          torch.where(N, n, big).min(-1).values)               # weakest literal
    s_min = torch.where(S["universal"], torch.ones_like(s_min), s_min)          # unused for universal
    d_false_glob = torch.sigmoid(bt + torch.log(1.0 - s_min * (1.0 - eps_bar) + eps) + (D - 1) * log1pe)

    d_glob = torch.where(S["universal"], d_true,
                         torch.where(S["contradictory"], d_false_glob, torch.maximum(d_true, d_false_glob)))
    return {
        "U": U, "A": Aact, "d_true": d_true, "d_false_glob": d_false_glob, "d_glob": d_glob,
        "D_glob_a": d_glob.reshape(A, K).sum(-1),                                # [A]
        "sP": torch.where(P, p, torch.zeros_like(p)), "sN": torch.where(N, n, torch.zeros_like(n)),
    }


def violated_strength(H, S, SC):
    """Strength (p or n) of the strongest violated literal of each clause at x -- independent of eps_bar."""
    ch = H["c_hard"].unsqueeze(1)                                               # [B,1,D]
    s_viol = SC["sP"] * (S["P"] & ~ch) + SC["sN"] * (S["N"] & ch)              # [B,C,D]
    return s_viol.max(-1).values                                                # [B,C]


def structural_part(c_soft, H, S, SC, eps_bar, eps=EPS, s_max=None):
    """State-wise and global Cleanliness certificates (no soft forward pass needed)."""
    B, D = c_soft.shape
    A, K = S["n_actions"], S["n_clauses"]
    eps_j = torch.minimum(c_soft, 1.0 - c_soft)
    dirty = (eps_j >= eps_bar) & S["rule_concepts"]
    clean = ~dirty.any(-1)                                                      # [B]

    if s_max is None:
        s_max = violated_strength(H, S, SC)
    d_false_S = torch.sigmoid(S["b_tilde"] + torch.log(1.0 - s_max * (1.0 - eps_bar) + eps)
                              + (D - 1) * math.log1p(eps))
    d_S = torch.where(H["T"], SC["d_true"] + torch.zeros_like(d_false_S), d_false_S)   # [B,C]
    D_S_a = d_S.reshape(B, A, K).sum(-1)

    valid = H["trig"] & (H["gamma"] > 0)
    cert_S = clean & valid & (pairwise_slack(H["s_hard"], H["a_star"], D_S_a) > 0)
    cert_G = clean & valid & (pairwise_slack(H["s_hard"], H["a_star"], SC["D_glob_a"]) > 0)
    # Delta <= Dbar must hold on every clean state (per clause)
    excess = (H["delta"] - d_S) * clean.unsqueeze(-1)
    return {"clean": clean, "cert_S": cert_S, "cert_G": cert_G, "dbar_excess": excess,
            "n_dirty_rule": dirty.sum(-1)}


# ============================================================================
# Main analysis
# ============================================================================

@torch.no_grad()
def run(args):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = pick_device(args.device, dtype)
    env_name, env_short = check_env_name(args.env_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device} | dtype: {args.dtype} | env: {env_name or '(unspecified)'}")

    model, config = load_lucid(args.model_path, device, dtype)
    model32 = None
    if dtype == torch.float64:
        model32, _ = load_lucid(args.model_path, device if device != "mps" else "cpu", torch.float32)

    tau = args.tau
    theta = args.theta
    cfg_theta = getattr(config, "hard_threshold", None)
    if cfg_theta is not None and abs(float(cfg_theta) - theta) > 1e-12:
        print(f"[NOTE] checkpoint config.hard_threshold={cfg_theta} differs from --hard_threshold={theta}; "
              f"using {theta}.")
    if tau < 0.5:
        print(f"[NOTE] threshold tau={tau} < 1/2: contradictory clauses possible; they are evaluated as False.")
    elif tau < 2.0 / 3.0:
        print(f"[NOTE] threshold tau={tau} is in [1/2, 2/3): majority guarantee holds; the confidence "
              f"guarantee (tau >= 2/3) holds only if the selectors are polarized (Corollary, part c).")
    thetas = sorted(set([theta] + [float(t) for t in args.theta_sweep]))
    eps_bar = args.clean_eps
    if not (0.0 < eps_bar <= min(theta, 1.0 - theta)):
        raise ValueError(f"--clean_eps must be in (0, min(theta, 1-theta)] = (0, {min(theta, 1 - theta):.3f}]")
    inv_lo, inv_hi = args.invariance_range

    S = extract_structure(model, tau)
    A, K, D = S["n_actions"], S["n_clauses"], S["D"]
    action_names = get_action_names(env_name, A)
    rstats = rule_stats(S)
    print(f"Model: D={D}, A={A}, clauses/action={K}, tau={tau}, theta={theta}")
    print(f"Rules: {rstats}")
    SC = structural_constants(S, eps_bar)
    eps_grid = sorted({float(e) for e in args.clean_eps_grid if 0.0 < e <= min(theta, 1.0 - theta)} | {eps_bar},
                      reverse=True)
    SC_grid = {e: (SC if e == eps_bar else structural_constants(S, e)) for e in eps_grid}
    grid_cnt = {e: {"clean": 0, "cert_S": 0, "cert_G": 0, "viol": 0} for e in eps_grid}
    sel_in_range = int((((S["p"] > inv_lo) & (S["p"] <= inv_hi)).sum() + ((S["n"] > inv_lo) & (S["n"] <= inv_hi)).sum()))

    features, teacher_all, pre_norm = load_data(args.data_path, args.split)
    N_all = len(features)
    if args.max_samples and args.max_samples < N_all:
        g = torch.Generator().manual_seed(args.seed)
        sel = torch.randperm(N_all, generator=g)[: args.max_samples].sort().values
        features = features[sel]
        teacher_all = teacher_all[sel] if teacher_all is not None else None
    Nn = len(features)
    if teacher_all is not None and int(teacher_all.max()) >= A:
        print(f"[WARN] teacher actions >= n_actions={A} found; clamped as in train_joint.py.")
        teacher_all = teacher_all.clamp(0, A - 1)
    print(f"Data: {Nn} states (split={args.split}, pre_normalized={pre_norm})")

    if args.fallback_action is not None:
        fallback = int(args.fallback_action)
    elif teacher_all is not None:
        fallback = int(torch.bincount(teacher_all, minlength=A).argmax())
    else:
        fallback = 0
        print("[WARN] No teacher actions in data; fallback action set to 0.")
    print(f"Fallback action: {fallback} ({action_names[fallback] if fallback < A else '?'})")

    # ---------------- accumulators ----------------
    per_state = {k: [] for k in [
        "gamma", "delta_star", "delta_other", "delta_total", "pair_slack",
        "dstar_true", "dstar_false", "dother_true", "dother_false", "rule_softness",
        "trig", "trig_nonuniv", "tie", "cert", "cert_simple", "agree",
        "pi_soft", "pi_hard", "teacher", "a_other",
        "clean", "cert_S", "cert_G", "n_dirty_rule", "theta_sensitive"]}
    nbins = args.hist_bins
    hist_all = torch.zeros(nbins, dtype=torch.float64)
    hist_rule = torch.zeros(nbins, dtype=torch.float64)
    n_clean_all = n_clean_rule = n_tot_all = n_tot_rule = 0
    n_near_theta_rule = 0
    fragile_false = torch.zeros(D, dtype=torch.float64, device=device)
    fragile_drag = torch.zeros(D, dtype=torch.float64, device=device)
    diag = {"true_clause_count": 0, "drag_active_sum": 0.0, "drag_inactive_sum": 0.0,
            "false_pos_count": 0, "false_pos_delta_sum": 0.0, "false_pos_lstar_sum": 0.0,
            "false_neg_count": 0, "false_neg_delta_sum": 0.0, "false_neg_lstar_sum": 0.0}
    checks = {"max_forward_diff": 0.0, "lemma1_violations": 0, "lemma1_max_excess": -float("inf"),
              "lemma2_violations": 0, "lemma2_max_excess": -float("inf"),
              "float32_soft_argmax_mismatch": 0,
              "dbar_violations": 0, "dbar_max_excess": -float("inf")}
    sweep = {t: None for t in thetas}

    rule_mask = S["rule_concepts"]
    universal = S["universal"]
    act_of_clause = S["clause_action"]
    tol = 1e-9 if dtype == torch.float64 else 1e-5

    for start in tqdm(range(0, Nn, args.batch_size), desc="Certifying"):
        xb = features[start:start + args.batch_size]
        tb = teacher_all[start:start + args.batch_size].to(device) if teacher_all is not None else None
        x = xb.to(device=device, dtype=dtype)
        x_norm = x if pre_norm else model.normalize_input(x)

        z_sparse, _ = model.sae.encode(x_norm)
        c_soft = model.bottleneck(model.normalize_z(z_sparse))          # [B,D]
        s_model = model.logic_layer(c_soft)                             # model's own forward

        soft = soft_part(c_soft, S)
        checks["max_forward_diff"] = max(checks["max_forward_diff"],
                                         float((soft["s_soft"] - s_model).abs().max()))
        if model32 is not None:
            x32 = xb.to(device=next(model32.parameters()).device)
            x32 = x32 if pre_norm else model32.normalize_input(x32)
            pi32 = model32(x32).argmax(-1).to(device)
            checks["float32_soft_argmax_mismatch"] += int((pi32 != soft["s_soft"].argmax(-1)).sum())

        # theta sweep (summary counts only)
        for t in thetas:
            Ht = hard_part(c_soft, soft, S, t, fallback)
            cnt = sweep_counts(Ht, tb)
            sweep[t] = cnt if sweep[t] is None else {k: sweep[t][k] + cnt[k] for k in cnt}
            if t == theta:
                H = Ht

        B = c_soft.shape[0]
        T = H["T"]
        agree = H["pi_soft"] == H["pi_hard"]

        # ---- self-checks
        l1 = H["lemma1_excess"]; l2 = H["lemma2_excess"]
        checks["lemma1_violations"] += int((l1 > tol).sum())
        checks["lemma1_max_excess"] = max(checks["lemma1_max_excess"], float(l1.max()))
        checks["lemma2_violations"] += int((l2 > tol).sum())
        checks["lemma2_max_excess"] = max(checks["lemma2_max_excess"], float(l2.max()))

        # ---- Cleanliness corollary (state-wise and global)
        s_max = violated_strength(H, S, SC)
        St = None
        for e in eps_grid:
            Se = structural_part(c_soft, H, S, SC_grid[e], e, s_max=s_max)
            grid_cnt[e]["clean"] += int(Se["clean"].sum())
            grid_cnt[e]["cert_S"] += int(Se["cert_S"].sum())
            grid_cnt[e]["cert_G"] += int(Se["cert_G"].sum())
            grid_cnt[e]["viol"] += int((Se["dbar_excess"] > tol).sum())
            if e == eps_bar:
                St = Se
        checks["dbar_violations"] += int((St["dbar_excess"] > tol).sum())
        checks["dbar_max_excess"] = max(checks["dbar_max_excess"], float(St["dbar_excess"].max()))
        # theta-invariance: a rule concept in (lo, hi] can flip c_hard for some theta in [lo, hi]
        theta_sens = (((c_soft > inv_lo) & (c_soft <= inv_hi)) & rule_mask).any(-1)

        # ---- Delta decomposition: true-clause drag vs false-clause residual
        d_true = (H["delta"] * T).reshape(B, A, K).sum(-1)
        d_false = (H["delta"] * ~T).reshape(B, A, K).sum(-1)
        a_s = H["a_star"].unsqueeze(-1); a_o = H["a_other"].unsqueeze(-1)

        # ---- per-state records
        eps_j = torch.minimum(c_soft, 1.0 - c_soft)
        rule_soft = (eps_j * rule_mask).max(-1).values if rule_mask.any() else torch.zeros(B, device=device, dtype=dtype)
        rec = {
            "gamma": H["gamma"], "delta_star": H["delta_star"], "delta_other": H["delta_other"],
            "delta_total": H["delta_total"], "pair_slack": H["pair_slack"],
            "dstar_true": d_true.gather(-1, a_s).squeeze(-1), "dstar_false": d_false.gather(-1, a_s).squeeze(-1),
            "dother_true": d_true.gather(-1, a_o).squeeze(-1), "dother_false": d_false.gather(-1, a_o).squeeze(-1),
            "rule_softness": rule_soft,
            "trig": H["trig"], "trig_nonuniv": (T & ~universal).any(-1), "tie": H["tie"],
            "cert": H["cert"], "cert_simple": H["cert_simple"], "agree": agree,
            "pi_soft": H["pi_soft"], "pi_hard": H["pi_hard"],
            "teacher": tb if tb is not None else torch.full((B,), -1, device=device),
            "a_other": H["a_other"],
            "clean": St["clean"], "cert_S": St["cert_S"], "cert_G": St["cert_G"],
            "n_dirty_rule": St["n_dirty_rule"], "theta_sensitive": theta_sens,
        }
        for k, v in rec.items():
            v = v.detach().cpu()
            per_state[k].append(v.numpy().astype(np.float32) if v.is_floating_point() else v.numpy())

        # ---- binarization quality
        cs = c_soft.detach().float().cpu()
        hist_all += torch.histc(cs, bins=nbins, min=0.0, max=1.0).double()
        clean = (cs < 0.05) | (cs > 0.95)
        n_clean_all += int(clean.sum()); n_tot_all += cs.numel()
        rm = rule_mask.cpu()
        if rm.any():
            cr = cs[:, rm]
            hist_rule += torch.histc(cr, bins=nbins, min=0.0, max=1.0).double()
            n_clean_rule += int(((cr < 0.05) | (cr > 0.95)).sum()); n_tot_rule += cr.numel()
            n_near_theta_rule += int(((cr - theta).abs() < args.near_theta).sum())

        # ---- mechanism diagnostics
        act = S["active"].unsqueeze(0)                                 # [1,C,D]
        L_act = (soft["log_l"] * act).sum(-1)                          # [B,C]
        L_inact = soft["L"] - L_act
        diag["true_clause_count"] += int(T.sum())
        diag["drag_active_sum"] += float((-L_act * T).sum())
        diag["drag_inactive_sum"] += float((-L_inact * T).sum())

        F_ = ~T
        jst = H["j_star"]
        P_at = S["P"].unsqueeze(0).expand(B, -1, -1).gather(-1, jst.unsqueeze(-1)).squeeze(-1)
        ch_at = H["c_hard"].gather(-1, jst)                            # [B,C]
        pos_type = F_ & P_at & ~ch_at                                  # positive literal violated (c <= theta)
        neg_type = F_ & ~pos_type                                      # negative literal violated (c > theta)
        for name, m in (("pos", pos_type), ("neg", neg_type)):
            diag[f"false_{name}_count"] += int(m.sum())
            diag[f"false_{name}_delta_sum"] += float((H["delta"] * m).sum())
            diag[f"false_{name}_lstar_sum"] += float((H["l_star"] * m).sum())

        # ---- fragile concepts (uncertified states only, actions a* and the max-Delta rival)
        unc = ~H["cert"]
        if unc.any():
            rel = (act_of_clause.unsqueeze(0) == H["a_star"].unsqueeze(-1)) | \
                  (act_of_clause.unsqueeze(0) == H["a_other"].unsqueeze(-1))   # [B,C]
            mf = F_ & rel & unc.unsqueeze(-1)
            fragile_false.index_add_(0, jst[mf], H["delta"][mf].double())
            mt = (T & rel & unc.unsqueeze(-1)).unsqueeze(-1)
            fragile_drag += (-soft["log_l"] * mt).sum((0, 1)).double()

        # ---- print a failure immediately if the theorem is violated
        bad = H["cert"] & ~agree
        if bad.any():
            print(f"\n[ERROR] Theorem violated on {int(bad.sum())} certified states in batch starting at {start}!")

    # ------------------------------------------------------------------ aggregate
    arr = {k: np.concatenate(v) for k, v in per_state.items()}
    trig = arr["trig"].astype(bool); cert = arr["cert"].astype(bool)
    cert_s = arr["cert_simple"].astype(bool); agree = arr["agree"].astype(bool)
    has_teacher = teacher_all is not None
    teacher = arr["teacher"]

    def pct(x):
        return float(np.mean(x) * 100.0) if len(x) else float("nan")

    def stats(x):
        if len(x) == 0:
            return {"mean": float("nan"), "median": float("nan"), "p5": float("nan"), "p95": float("nan")}
        return {"mean": float(np.mean(x)), "median": float(np.median(x)),
                "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95))}

    coverage = pct(cert)
    coverage_simple = pct(cert_s)
    overall_agree = pct(agree)
    covered_agree = pct(agree[cert])
    covered_agree_simple = pct(agree[cert_s])
    uncov_disagree = pct(~agree[~cert])
    theorem_violations = int((cert & ~agree).sum())

    summary = {
        "env_name": env_name, "env_short": env_short,
        "model_path": args.model_path, "data_path": args.data_path, "split": args.split,
        "n_states": int(len(agree)), "tau": tau, "theta": theta, "eps": EPS, "b_max": B_MAX,
        "D": D, "n_actions": A, "n_clauses_per_action": K,
        "fallback_action": fallback, "action_names": action_names,
        "rules": rstats,
        "binarization_cleanliness_all": 100.0 * n_clean_all / max(n_tot_all, 1),
        "binarization_cleanliness_rule": 100.0 * n_clean_rule / max(n_tot_rule, 1) if n_tot_rule else float("nan"),
        "rule_concepts_near_theta_pct": 100.0 * n_near_theta_rule / max(n_tot_rule, 1) if n_tot_rule else float("nan"),
        "trigger_rate": pct(trig),
        "trigger_rate_nonuniversal": pct(arr["trig_nonuniv"].astype(bool)),
        "tie_rate": pct(arr["tie"].astype(bool)),
        "gamma_on_trig": stats(arr["gamma"][trig]),
        "delta_total_on_trig": stats(arr["delta_total"][trig]),
        "fidelity_coverage_pairwise": coverage,
        "fidelity_coverage_simple": coverage_simple,
        "covered_agreement_pairwise": covered_agree,
        "covered_agreement_simple": covered_agree_simple,
        "overall_soft_hard_agreement": overall_agree,
        "uncovered_disagreement_rate": uncov_disagree,
        "theorem_bound_holds": bool((100.0 - overall_agree) <= (100.0 - coverage) + 1e-9),
        "checks": {**checks, "theorem_violations": theorem_violations,
                   "float32_soft_argmax_mismatch_pct": 100.0 * checks["float32_soft_argmax_mismatch"] / max(len(agree), 1)
                   if model32 is not None else None},
        "delta_decomposition": {
            "mean_drag_active_per_true_clause": diag["drag_active_sum"] / max(diag["true_clause_count"], 1),
            "mean_drag_inactive_per_true_clause": diag["drag_inactive_sum"] / max(diag["true_clause_count"], 1),
            "false_clauses_pos_violation": diag["false_pos_count"],
            "false_clauses_neg_violation": diag["false_neg_count"],
            "mean_delta_false_pos": diag["false_pos_delta_sum"] / max(diag["false_pos_count"], 1),
            "mean_delta_false_neg": diag["false_neg_delta_sum"] / max(diag["false_neg_count"], 1),
            "mean_lstar_pos": diag["false_pos_lstar_sum"] / max(diag["false_pos_count"], 1),
            "mean_lstar_neg": diag["false_neg_lstar_sum"] / max(diag["false_neg_count"], 1),
            "structural_bound_lstar_pos": 1.0 - tau * (1.0 - theta),   # Lemma (violated literals)
            "structural_bound_lstar_neg": 1.0 - tau * theta,
            "mean_dstar_true_on_trig": float(np.mean(arr["dstar_true"][trig])) if trig.any() else float("nan"),
            "mean_dstar_false_on_trig": float(np.mean(arr["dstar_false"][trig])) if trig.any() else float("nan"),
            "mean_dother_true_on_trig": float(np.mean(arr["dother_true"][trig])) if trig.any() else float("nan"),
            "mean_dother_false_on_trig": float(np.mean(arr["dother_false"][trig])) if trig.any() else float("nan"),
        },
    }

    if has_teacher:
        af_soft = pct(arr["pi_soft"] == teacher)
        af_hard = pct(arr["pi_hard"] == teacher)
        imit = 100.0 - af_soft
        disc = 100.0 - overall_agree
        summary["teacher"] = {
            "AF_soft": af_soft, "AF_hard": af_hard,
            "hard_error": 100.0 - af_hard,
            "imitation_error": imit,
            "discretization_error": disc,
            "triangle_middle": imit + disc,
            "triangle_bound": imit + (100.0 - coverage),
            "triangle_holds": bool((100.0 - af_hard) <= imit + disc + 1e-9 and imit + disc <= imit + (100.0 - coverage) + 1e-9),
        }

    clean = arr["clean"].astype(bool)
    cS = arr["cert_S"].astype(bool); cG = arr["cert_G"].astype(bool)
    tie = arr["tie"].astype(bool)
    pos_g = arr["gamma"][trig & ~tie]
    real = ~S["contradictory"] & ~S["universal"]
    Dg = SC["D_glob_a"].detach().cpu().numpy().astype(float)
    Dg_pair = float(np.max(Dg[:, None] + Dg[None, :] + np.diag(np.full(len(Dg), -np.inf)))) if len(Dg) > 1 else float(Dg[0])
    gmin = float(pos_g.min()) if len(pos_g) else float("nan")
    U_np = SC["U"].detach().cpu().numpy(); A_np = SC["A"].detach().cpu().numpy()
    real_np = real.detach().cpu().numpy().astype(bool)
    summary["corollary"] = {
        "clean_eps": eps_bar,
        "clean_state_rate": pct(clean),
        "actual_disagreement": 100.0 - overall_agree,
        "bound_theorem": 100.0 - coverage,
        "bound_statewise": 100.0 - pct(cS),
        "bound_global": 100.0 - pct(cG),
        "not_clean": 100.0 - pct(clean),
        "clean_but_tie_or_fallback": pct(clean & ~(trig & ~tie)),
        "clean_valid_uncertified_statewise": pct(clean & trig & ~tie & ~cS),
        "clean_valid_uncertified_global": pct(clean & trig & ~tie & ~cG),
        "mean_dirty_rule_concepts_per_state": float(np.mean(arr["n_dirty_rule"])),
        "structural_drag_U": {"mean": float(U_np[real_np].mean()) if real_np.any() else float("nan"),
                              "max": float(U_np[real_np].max()) if real_np.any() else float("nan"),
                              "universal_mean": float(U_np[~real_np].mean()) if (~real_np).any() else float("nan")},
        "active_drag_A": {"mean": float(A_np[real_np].mean()) if real_np.any() else float("nan"),
                          "max": float(A_np[real_np].max()) if real_np.any() else float("nan")},
        "Dbar_global_per_action": Dg.tolist(),
        "Dbar_global_pair_max": Dg_pair,
        "gamma_min_positive_empirical": gmin,
        "global_condition_holds_on_data": bool(len(pos_g) and gmin > Dg_pair),
        "checks": {"dbar_violations": checks["dbar_violations"], "dbar_max_excess": checks["dbar_max_excess"],
                   "certS_disagree": int((cS & ~agree).sum()), "certG_disagree": int((cG & ~agree).sum()),
                   "certG_not_in_certS": int((cG & ~cS).sum()), "certS_not_in_cert": int((cS & ~cert).sum())},
    }
    grid = []
    for e in eps_grid:
        g_ = grid_cnt[e]
        grid.append({"eps_bar": e, "clean_state_rate": 100.0 * g_["clean"] / Nn,
                     "bound_statewise": 100.0 - 100.0 * g_["cert_S"] / Nn,
                     "bound_global": 100.0 - 100.0 * g_["cert_G"] / Nn, "dbar_violations": g_["viol"]})
    best_S = min(grid, key=lambda r: r["bound_statewise"])
    best_G = min(grid, key=lambda r: r["bound_global"])
    summary["corollary"]["eps_grid"] = grid
    summary["corollary"]["best_statewise"] = {"eps_bar": best_S["eps_bar"], "bound": best_S["bound_statewise"]}
    summary["corollary"]["best_global"] = {"eps_bar": best_G["eps_bar"], "bound": best_G["bound_global"]}
    summary["corollary"]["checks"]["dbar_violations_grid"] = int(sum(r["dbar_violations"] for r in grid))
    summary["corollary"]["e_minus_bmax"] = math.exp(-B_MAX)

    summary["invariance"] = {
        "range": [inv_lo, inv_hi],
        "selectors_in_range": sel_in_range,
        "tau_invariant": sel_in_range == 0,
        "states_theta_sensitive": int(arr["theta_sensitive"].astype(bool).sum()),
        "states_theta_sensitive_pct": pct(arr["theta_sensitive"].astype(bool)),
        "theta_invariant": int(arr["theta_sensitive"].astype(bool).sum()) == 0,
    }

    tab = []
    for t in thetas:
        c = sweep[t]
        n = max(c["n"], 1)
        tab.append({"theta": t, "coverage_pairwise": 100.0 * c["cert"] / n,
                    "coverage_simple": 100.0 * c["cert_simple"] / n,
                    "agreement": 100.0 * c["agree"] / n, "trigger_rate": 100.0 * c["trig"] / n,
                    "tie_rate": 100.0 * c["tie"] / n, "violations": c["violations"],
                    "AF_hard": 100.0 * c["hard_teacher"] / n if has_teacher else None})
    summary["theta_sweep"] = tab

    top = torch.topk(fragile_false + fragile_drag, k=min(args.top_fragile, D))
    summary["fragile_concepts"] = [
        {"concept": int(j), "total": float(v), "false_clause_delta": float(fragile_false[j]),
         "true_clause_drag": float(fragile_drag[j]), "in_rules": bool(rule_mask[j])}
        for j, v in zip(top.indices.tolist(), top.values.tolist()) if v > 0]

    # ------------------------------------------------------------------ report
    print_report(summary)

    # ------------------------------------------------------------------ save
    edges = np.linspace(0.0, 1.0, nbins + 1)
    np.savez_compressed(
        os.path.join(args.output_dir, "theorem2_analytics.npz"),
        **arr,
        hist_edges=edges, hist_all=hist_all.numpy(), hist_rule=hist_rule.numpy(),
        sweep_theta=np.array([r["theta"] for r in tab]),
        sweep_cov_pairwise=np.array([r["coverage_pairwise"] for r in tab]),
        sweep_cov_simple=np.array([r["coverage_simple"] for r in tab]),
        sweep_agreement=np.array([r["agreement"] for r in tab]),
        sweep_af_hard=np.array([r["AF_hard"] if r["AF_hard"] is not None else np.nan for r in tab]),
        sweep_tie=np.array([r["tie_rate"] for r in tab]),
        fragile_false=fragile_false.cpu().numpy(), fragile_drag=fragile_drag.cpu().numpy(),
        rule_concepts=rule_mask.cpu().numpy(),
    )
    with open(os.path.join(args.output_dir, "theorem2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "table5_row.tex"), "w") as f:
        f.write(table_row(summary) + "\n")
    print(f"\nSaved: {os.path.join(args.output_dir, 'theorem2_analytics.npz')}")
    print(f"       {os.path.join(args.output_dir, 'theorem2_summary.json')}")
    print(f"       {os.path.join(args.output_dir, 'table5_row.tex')}")

    cc = summary["corollary"]["checks"]
    ok = (theorem_violations == 0 and checks["lemma1_violations"] == 0 and checks["lemma2_violations"] == 0
          and cc["dbar_violations"] == 0 and cc["dbar_violations_grid"] == 0
          and cc["certS_disagree"] == 0 and cc["certG_disagree"] == 0
          and checks["max_forward_diff"] < (1e-6 if dtype == torch.float64 else 1e-3))
    if not ok:
        print("\n[FAIL] At least one self-check failed -- do not report these numbers before investigating.")
        sys.exit(2)
    return summary


def table_row(s):
    """Main Table 5 row: actual soft/hard disagreement vs the bounds (all in %)."""
    c = s["corollary"]
    clean_at = next(r["clean_state_rate"] for r in c["eps_grid"] if r["eps_bar"] == c["best_statewise"]["eps_bar"])
    return (f"\\textbf{{{s['env_short']}}} & {clean_at:.2f} & {s['tie_rate']:.2f} & "
            f"{c['actual_disagreement']:.2f} & {c['bound_theorem']:.2f} & "
            f"{c['best_statewise']['bound']:.2f} ({c['best_statewise']['eps_bar']:g}) & "
            f"{c['best_global']['bound']:.2f} ({c['best_global']['eps_bar']:g}) \\\\")


def print_report(s):
    line = "=" * 72
    print(f"\n{line}\nTHEOREM 1 (Soft--Hard Fidelity) -- {s['env_short']}  "
          f"[threshold tau={s['tau']}, hard_threshold theta={s['theta']}, N={s['n_states']}]\n{line}")
    r = s["rules"]
    print(f"Rules: {r['n_nonuniversal_clauses']} clauses, {r['n_literals']} literals, {r['n_concepts']} concepts "
          f"(+{r['n_universal_clauses']} universal, {r['n_contradictory_clauses']} contradictory)")
    print(f"Binarization cleanliness (all / rule concepts): {s['binarization_cleanliness_all']:.2f}% / "
          f"{s['binarization_cleanliness_rule']:.2f}%   | rule activations within +-near of theta: "
          f"{s['rule_concepts_near_theta_pct']:.3f}%")
    print(f"Trigger rate: {s['trigger_rate']:.2f}% (non-universal: {s['trigger_rate_nonuniversal']:.2f}%) | "
          f"ties (gamma=0): {s['tie_rate']:.2f}%")
    g, d = s["gamma_on_trig"], s["delta_total_on_trig"]
    print(f"gamma(x)       on X_trig | mean {g['mean']:.3f} | median {g['median']:.3f} | p5 {g['p5']:.3f} | p95 {g['p95']:.3f}")
    print(f"Delta_total(x) on X_trig | mean {d['mean']:.3f} | median {d['median']:.3f} | p5 {d['p5']:.3f} | p95 {d['p95']:.3f}")
    print(line)
    print(f"Fidelity coverage  Pr[X_cert] (pairwise, theorem)  : {s['fidelity_coverage_pairwise']:.2f}%")
    print(f"Fidelity coverage  (simple: Delta_total < gamma)   : {s['fidelity_coverage_simple']:.2f}%")
    print(f"Covered agreement  (pairwise / simple)             : {s['covered_agreement_pairwise']:.2f}% / "
          f"{s['covered_agreement_simple']:.2f}%")
    print(f"Overall soft-hard agreement                        : {s['overall_soft_hard_agreement']:.2f}%")
    print(f"Uncovered disagreement rate                        : {s['uncovered_disagreement_rate']:.2f}%")
    print(f"Pr[soft!=hard] <= 1 - Pr[X_cert]                   : "
          f"{100 - s['overall_soft_hard_agreement']:.2f}% <= {100 - s['fidelity_coverage_pairwise']:.2f}%  "
          f"-> {'OK' if s['theorem_bound_holds'] else 'VIOLATED'}")
    if "teacher" in s:
        t = s["teacher"]
        print(line)
        print(f"Teacher: AF_soft {t['AF_soft']:.2f}% | AF_hard {t['AF_hard']:.2f}%")
        print(f"Triangle: 1-AF_hard = {t['hard_error']:.2f} <= imit {t['imitation_error']:.2f} + disc "
              f"{t['discretization_error']:.2f} = {t['triangle_middle']:.2f} <= imit + (1-cov) = "
              f"{t['triangle_bound']:.2f}  -> {'OK' if t['triangle_holds'] else 'VIOLATED'}")
    dd = s["delta_decomposition"]
    print(line)
    print("Where does Delta come from? (on X_trig)")
    print(f"  Delta_a*   = true-clause drag {dd['mean_dstar_true_on_trig']:.3f} + false-clause residual {dd['mean_dstar_false_on_trig']:.3f}")
    print(f"  Delta_rival= true-clause drag {dd['mean_dother_true_on_trig']:.3f} + false-clause residual {dd['mean_dother_false_on_trig']:.3f}")
    print(f"  per true clause: -L_active {dd['mean_drag_active_per_true_clause']:.4f} | "
          f"-L_ignored (structural drag) {dd['mean_drag_inactive_per_true_clause']:.4f}")
    print(f"  false clauses by j*: positive-violated (c<=theta) n={dd['false_clauses_pos_violation']}, "
          f"mean l*={dd['mean_lstar_pos']:.3f} (worst case < {dd['structural_bound_lstar_pos']:.3f}), "
          f"mean Delta={dd['mean_delta_false_pos']:.4f}")
    print(f"                       negative-violated (c>theta)  n={dd['false_clauses_neg_violation']}, "
          f"mean l*={dd['mean_lstar_neg']:.3f} (worst case < {dd['structural_bound_lstar_neg']:.3f}), "
          f"mean Delta={dd['mean_delta_false_neg']:.4f}")
    print(line)
    print(f"hard_threshold (theta) sweep, threshold tau={s['tau']} fixed:")
    print(f"  {'theta':>6} | {'cov_pair':>8} | {'cov_simp':>8} | {'agree':>7} | {'AF_hard':>7} | {'ties':>6} | viol")
    for r in s["theta_sweep"]:
        afh = f"{r['AF_hard']:.2f}" if r["AF_hard"] is not None else "  -- "
        print(f"  {r['theta']:>6.3f} | {r['coverage_pairwise']:>8.2f} | {r['coverage_simple']:>8.2f} | "
              f"{r['agreement']:>7.2f} | {afh:>7} | {r['tie_rate']:>6.2f} | {r['violations']}")
    if s["fragile_concepts"]:
        print(line)
        print("Top concepts contributing to Delta on uncertified states:")
        for fc in s["fragile_concepts"][:10]:
            print(f"  f_{fc['concept']:<4} total {fc['total']:.2f} (false-clause {fc['false_clause_delta']:.2f}, "
                  f"true-clause drag {fc['true_clause_drag']:.2f}){'' if fc['in_rules'] else '  [not in rules]'}")
    co = s["corollary"]
    print(line)
    print(f"CLEANLINESS CERTIFICATE (Corollary; eps_bar={co['clean_eps']}, no soft forward pass needed)")
    print(f"  eps_bar-clean states (all rule concepts clean): {co['clean_state_rate']:.2f}%  "
          f"| mean #unclean rule concepts/state: {co['mean_dirty_rule_concepts_per_state']:.3f}")
    print(f"  structural drag U per clause (ignored literals): mean {co['structural_drag_U']['mean']:.4f}, "
          f"max {co['structural_drag_U']['max']:.4f} | active drag A: mean {co['active_drag_A']['mean']:.4f}, "
          f"max {co['active_drag_A']['max']:.4f}")
    print(f"  Dbar^G_a (global, data-independent): max pair {co['Dbar_global_pair_max']:.4f} vs empirical "
          f"min positive margin {co['gamma_min_positive_empirical']:.4f} -> global condition "
          f"{'HOLDS' if co['global_condition_holds_on_data'] else 'fails'} on this data")
    print(f"  Pr[soft != hard] actual              : {co['actual_disagreement']:.3f}%")
    print(f"  bound, Theorem 1 (1 - coverage)      : {co['bound_theorem']:.3f}%")
    print(f"  bound, Corollary state-wise          : {co['bound_statewise']:.3f}%  = not clean {co['not_clean']:.3f}"
          f" + clean&tie/fallback {co['clean_but_tie_or_fallback']:.3f} + clean&uncertified "
          f"{co['clean_valid_uncertified_statewise']:.3f}")
    print(f"  bound, Corollary global              : {co['bound_global']:.3f}%  (clean&uncertified "
          f"{co['clean_valid_uncertified_global']:.3f})")
    print(f"  eps_bar grid (min over a fixed grid of valid bounds is a valid bound; exp(-b_max) = "
          f"{co['e_minus_bmax']:.4f}):")
    print(f"    {'eps_bar':>8} | {'clean %':>8} | {'state-wise':>10} | {'global':>8}")
    for r in co["eps_grid"]:
        print(f"    {r['eps_bar']:>8.4g} | {r['clean_state_rate']:>8.2f} | {r['bound_statewise']:>10.3f} | "
              f"{r['bound_global']:>8.3f}")
    print(f"  best: state-wise {co['best_statewise']['bound']:.3f}% (eps_bar={co['best_statewise']['eps_bar']:g}), "
          f"global {co['best_global']['bound']:.3f}% (eps_bar={co['best_global']['eps_bar']:g})")
    iv = s["invariance"]
    print(line)
    print(f"INVARIANCE over [{iv['range'][0]}, {iv['range'][1]}]:")
    print(f"  selectors (p or n) inside the range: {iv['selectors_in_range']} -> rule set "
          f"{'IDENTICAL' if iv['tau_invariant'] else 'may change'} for every threshold tau in the range")
    print(f"  states with a rule-concept activation inside the range: {iv['states_theta_sensitive']} "
          f"({iv['states_theta_sensitive_pct']:.3f}%) -> hard policy "
          f"{'IDENTICAL' if iv['theta_invariant'] else 'can differ only on these states'} for hard_threshold theta in the range")
    c = s["checks"]
    cc = co["checks"]
    print(line)
    print("Self-checks:")
    print(f"  max |s_soft(reimpl) - s_soft(model.forward)| = {c['max_forward_diff']:.2e}")
    print(f"  Lemma 1 violations: {c['lemma1_violations']} (max excess {c['lemma1_max_excess']:.2e})")
    print(f"  Lemma 2 violations: {c['lemma2_violations']} (max excess {c['lemma2_max_excess']:.2e})")
    print(f"  Theorem violations (certified but soft != hard): {c['theorem_violations']}")
    print(f"  Corollary: Delta > Dbar on clean states: {cc['dbar_violations']} (max excess {cc['dbar_max_excess']:.2e}) | "
          f"certified (state-wise/global) but soft != hard: {cc['certS_disagree']}/{cc['certG_disagree']} | "
          f"nesting violations cert_G!<=cert_S: {cc['certG_not_in_certS']}, cert_S!<=cert: {cc['certS_not_in_cert']}")
    if c.get("float32_soft_argmax_mismatch_pct") is not None:
        print(f"  float32 vs float64 soft argmax mismatch: {c['float32_soft_argmax_mismatch_pct']:.4f}%")
    print(line)
    print("Table 5 row:\n  " + table_row(s))


def parse_args():
    ap = argparse.ArgumentParser(description="Verify the Soft-Hard Fidelity theorem (Table 5).")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--data_path", type=str, required=True,
                    help="Held-out rollout .pt with 'features' (raw) and 'actions' (teacher), "
                         "or training_data.pt (pre-normalized) together with --split val")
    ap.add_argument("--output_dir", type=str, default="./experiments/theorem/results")
    ap.add_argument("--env_name", type=str, default="",
                    help="Label only, e.g. PongNoFrameskip-v4, BoxingNoFrameskip-v4")
    ap.add_argument("--threshold", "--tau", dest="tau", type=float, default=0.66,
                    help="tau: selector threshold deciding p/n/ignore for the extracted rules "
                         "(P = {p > tau}, N = {n > tau}); same meaning as train_joint.py --threshold")
    ap.add_argument("--hard_threshold", "--theta", dest="theta", type=float, default=0.66,
                    help="theta: concept threshold for HARD inference, c_hard = 1[c_soft > theta]; "
                         "same meaning as train_joint.py --hard_threshold")
    ap.add_argument("--hard_threshold_sweep", "--theta_sweep", dest="theta_sweep", type=float, nargs="*",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9],
                    help="Extra theta values for the sensitivity sweep (pass nothing to disable)")
    ap.add_argument("--fallback_action", type=int, default=None,
                    help="Default: most frequent teacher action in --data_path")
    ap.add_argument("--split", type=str, default="all", choices=["all", "val", "train"])
    ap.add_argument("--max_samples", type=int, default=0, help="Random subsample (0 = all)")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Memory ~ 4 * B * (#clauses) * D * 8 bytes in float64")
    ap.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu (mps: float32 only)")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    ap.add_argument("--hist_bins", type=int, default=200)
    ap.add_argument("--near_theta", type=float, default=0.05)
    ap.add_argument("--clean_eps", type=float, default=0.05,
                    help="eps_bar of the Cleanliness corollary: a state is clean if every rule concept has "
                         "min(c, 1-c) < eps_bar (same 0.05 as the Binarization Cleanliness column)")
    ap.add_argument("--clean_eps_grid", type=float, nargs="*",
                    default=[0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3, 1e-4, 1e-5],
                    help="eps_bar values for the Corollary; the reported bound is the minimum over this fixed grid "
                         "(values > min(theta, 1-theta) are dropped)")
    ap.add_argument("--invariance_range", type=float, nargs=2, default=[0.5, 0.9],
                    help="Range [lo, hi] for the tau / theta invariance counts")
    ap.add_argument("--top_fragile", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
