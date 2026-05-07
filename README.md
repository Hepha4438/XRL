# LUCID: Automated Concept Discovery and Logic Rule Induction for XRL

**LUCID** (Logic-based Understanding through Concept Induction and Decoding) is a framework designed to extract verifiable logic rules from a frozen PPO policy. It combines **Sparse Autoencoders (SAE)** for concept discovery and **Product T-Norm Neural Logic** for rule induction.

LUCID achieves **100% success rate** on MiniGrid environments and provides human-readable explanations in Disjunctive Normal Form (DNF), grounded with semantic labels using Vision-Language Models (**Gemini 2.5 Pro**).

---

## Architecture

LUCID employs a two-stage training procedure to prevent gradient interference and ensure stable concept learning:

1.  **Feature Extraction & SAE Training**
    * Collect activations from the frozen PPO Encoder.
    * Train a Sparse Autoencoder with **TopK** activation ($k$) to discover monosemantic concepts.
2.  **Logic Induction & Semantic Grounding**
    * **Binarization Bottleneck:** Maps continuous SAE activations to Boolean values (0/1).
    * **Product T-Norm Logic Layer:** Induces DNF rules (AND-OR logic) to mimic the PPO's behavior.
    * **VLM Grounding:** Automatically labels SAE features using Gemini 2.5 Pro by triangulating Integrated Gradients heatmaps with DNF logical context.

```
Raw observation (MiniGrid)
        │
        ▼
PPO CNN feature extractor  (frozen — never trained)
        │  (N, d)
        ▼
Normalize  →  (x - mean) / std
        │  (N, d)
        ▼
┌─────────────────────────────────────────────────┐
│  SAE Pre-training (recon only)                  │
│                                                 │
│  Sparse Autoencoder (SAE)                       │
│    encoder: Linear(d → D) + TopK(k=50)          │
│    decoder: Linear(D → d) [unit-norm cols]      │
│    loss: MSE reconstruction + L1 sparsity       │
│                                                 │
│  ► SAE frozen after convergence                 │
│  ► Activation stats computed (fixed buffers)    │
└─────────────────────────────────────────────────┘
        │  sparse z  (N, D),  only k active per sample
        ▼
Fixed Normalization  →  (z - z_mean) / z_std
        │  per-feature stats from non-zero activations (buffers, not learned)
        ▼
┌─────────────────────────────────────────────────┐
│  Logic Training (SAE frozen)                    │
│                                                 │
│  Sigmoid Bottleneck                             │
│    output_i = sigmoid(α_i · (z_i − β_i))        │
│    α, β learnable per feature                   │
│    bimodality loss pushes outputs toward {0, 1} │
│            │                                    │
│            ▼                                    │
│  Product T-Norm Logic Layer (DNF)               │
│    For each action:  OR(clause_1, ..., clause_n)│
│    Each clause:      AND over soft literals     │
│    literal = p·f + n·(1−f) + (1−p−n)·1          │
│    p, n from 3-way softmax → differentiable     │
└─────────────────────────────────────────────────┘
        │  (N, num_actions)  action logits
        ▼
argmax  →  action
```

---

## Pipeline Execution

Below is the complete execution pipeline for the MiniGrid-DoorKey-6x6-v0 environment. For the commands to run other environments (such as Dynamic Obstacles or Pixel CartPole), please refer to the cmd file included in this repository.

### Step 0: Pre-trained Teacher Policy
Firstly, before running the LUCID pipeline, you must have a pre-trained Reinforcement Learning agent (e.g., PPO) that has already mastered the target environment. This agent acts as the "Teacher" from which we will extract logic rules. 

If you don't have a pre-trained model, you can train one using the provided baseline script:

```bash
python train_doorkey_6x6.py
```
### Step 1: Rollout data
Collect rollout data from the PPO agent and compute initial normalization statistics.
```bash
python feature_collect.py \
    --model_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_episodes 2000 \
    --save_dir ./stage1_outputs_doorkey
```
### Step 2: Joint SAE & Logic Training
LUCID trains the neuro-symbolic pipeline in two stages. First, the SAE is pre-trained for reconstruction only (Stage 2A) to stabilize the feature space. Then, the SAE is frozen, and the Product T-Norm logic layer is trained (Stage 2B) to induce DNF rules based on those stable features.

```bash
python train_joint.py \
    --features_path ./stage1_outputs_doorkey/collected_data.pt \
    --stage1_path ./stage1_outputs_doorkey/stage1_outputs.pt \
    --hidden_dim 300 \
    --k 50 \
    --n_clauses_per_action 20 \
    --n_epochs 600 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --seed 42 \
    --threshold 0.5 \
    --entropy_weight 0.06 \
    --bimodal_ramp 120 \
    --save_dir ./outputs/lucid_model_doorkey
```
**Key Training Dynamics:**
* **A:**: Minimizes MSE reconstruction loss to ensure the SAE captures all relevant PPO features before logic induction begins.
* **B:**: It optimizes Cross-Entropy for action imitation while applying Bimodal and Entropy losses to ensure the final rules are discrete and concise.

---

### Step 3: Evaluation & Fidelity Metrics
Evaluate the performance of the induced rules against the original PPO teacher. This script calculates the **Soft-Hard Fidelity** (agreement between the logic layer and the PPO) and the task Success Rate.

```bash
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_doorkey/collected_data.pt \
    --threshold 0.5 \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --ppo_path ./ppo_doorkey_6x6.zip \
    --episodes 1000 \
    --multi-seed
```

### Step 4: Semantic Concept Grounding (VLM)
This step assigns human-readable labels to the discovered SAE concepts. It uses **Gemini 2.5 Pro** as a Vision-Language Model (VLM) to analyze Integrated Gradients heatmaps and triangulate them with the logical context of the DNF rules.

1. **Generate Heatmaps:** Visualize which pixels activate specific SAE features.
2. **Contrastive Prompting:** Send the heatmaps and rule context to Gemini 2.5 Pro to induce semantic labels (e.g., "Key in front", "Door is locked").

```bash
python auto_label_concepts.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --ppo_path ./ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --episodes 100 \
    --top_k 8

python gemini_concept_labeler.py \
    --img_dir ./concept_grounding/MiniGrid_DoorKey_6x6_v0 \
    --env_name "MiniGrid DoorKey 6x6" \
    --top_k 8 \
    --rules_path ./outputs/lucid_model_doorkey/learned_rules.json \
    --model_name gemini-2.5-pro \
    --api_key YOUR_API_KEY
```

### Step 5: Verify Theorem
This step empirically validates the **Soft-Hard Fidelity Guarantee (Theorem)** of the LUCID framework. The process analyzes the action margin $\gamma(x)$ against the conceptual perturbation bound $\Delta_{\text{total}}(x)$ to verify that the extracted discrete logic perfectly matches the continuous soft policy within the mathematically defined "safe zone".

The validation pipeline consists of three sequential stages:
1. **Data Collection**: Generate a massive held-out offline dataset to rigorously evaluate the theoretical error bounds on unseen states.
2. **Fidelity Verification:** Compute the exact conceptual perturbations and action margins, formally separating the dataset into covered (safe) and uncovered (fragile) zones.
3. **Visualization & Analytics:** Generate statistical plots (scatter plots of the bounds and binarization histograms) to report Binarization Cleanliness, Fidelity Coverage, and the $100\%$ Covered Agreement.

```bash
python collect_with_observations.py \
    --model_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_episodes 2000 \
    --save_path held_out_doorkey_data.pt \
    --seed 42

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --data_path ./held_out_doorkey_data.pt \
    --output_dir ./experiments/theorem/results_doorkey

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_doorkey/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_doorkey/figures
```

### Step 6: Ablation Studies (Threshold Sensitivity)
Analyze how the binarization threshold $\tau$ affects the model's performance on held-out data. This helps verify the "Schelling optimality" of $\tau = 0.5$ as discussed in Theorem 1.

```bash
python experiments/tau/run_tau_ablation.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --features_path ./held_out_doorkey_data.pt \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --output_dir ./experiments/tau/results_doorkey \
    --episodes 1000
```

### Step 7: Baseline Comparisons (Decision Trees)
Compare LUCID's performance against standard interpretable baselines including SA-DT (State-Action Decision Tree), VIPER (DAgger-based Decision Tree), and Soft-DT.

```bash
python experiments/dt/sa_dt_baseline.py \
    --ppo_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --data_path stage1_outputs_doorkey/collected_data.pt \
    --save_dir ./experiments/dt/results \
    --n_eval_episodes 1000 \
    --max_depth 4 \
    --multi-seed

python experiments/dt/viper_baseline.py \
    --ppo_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --max_depth 8 \
    --save_dir experiments/dt/results/doorkey_6x6 \
    --n_dagger_iters 75 \
    --episodes_per_iter 50 \
    --max_steps 100 \
    --n_eval_episodes 1000 \
    --multi-seed \
    --min_samples_leaf 1

python experiments/soft_dt/soft_dt_baseline.py \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_eval_episodes 1000 \
    --max_depth 4 \
    --multi-seed \
    --epochs 1000
```
---

### Key Results & Statistics

| Environment / Metric | PPO | LUCID (Soft) | LUCID (Hard) | SA-DT | VIPER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DoorKey-6x6** *(1000 ep)* | | | | | |
| Score | 0.9658 | **0.9658** | 0.9652 | 0.9511 | 0.0000 |
| Success Rate (SR) | 100.0% | **100.0%** | **100.0%** | 98.50% | 0.00% |
| Action Fidelity | - | **99.98%** | 98.87% | 99.21% | 96.53% |
| **PixelCartPole** *(200 ep)* | | | | | |
| Score | 500.0 | **482.81** | 442.33 | 419.20 | 398.11 |
| Action Fidelity | - | **99.60%** | 95.81% | 91.75% | 92.45% |
| **Dynamic-Obs-5x5** *(1000 ep)* | | | | | |
| Score | 0.8931 | 0.8931 | 0.8931 | 0.8931 | 0.8951 |
| Success Rate (SR) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Action Fidelity | - | 100.0% | 100.0% | 100.0% | 99.95% |

*Note: For Pixel-CartPole, we report Score/Fidelity instead of Success Rate as it is a continuous balancing task (maintaining the pole) rather than a goal-oriented mission.*

## Main Arguments & Hyperparameters

| Argument | Description |
| :--- | :--- |
| `--entropy_weight` | Controls rule sparsity (L0 penalty). Higher values lead to shorter rules. |
| `--bimodal_ramp` | Epochs to ramp up the Bimodal Loss, pushing outputs toward discrete {0, 1}. |
| `--k` (TopK) | Sparsity constraint; the number of active SAE features per sample. |
| `--save_training_data` | Saves intermediate activations for post-hoc VLM semantic labeling. |

---

## Project Structure

```text
.
├── feature_space_analysis.py      # Step 1: Data collection & Subspace Analysis
├── train_joint.py                 # Step 2: Joint Logic & SAE training (LUCID)
├── auto_label_concepts.py         # Generates feature heatmaps for VLM grounding
├── gemini_concept_labeler.py      # Uses Gemini to infer semantic labels for concepts
└── experiments/
    ├── lucid/                     # Fidelity and metrics evaluation
    ├── dt/                        # Decision Tree baselines (VIPER, SA-DT)
    ├── soft_dt/                   # Soft Decision Tree baseline
    ├── tau/                       # Threshold (τ) ablation studies
    └── theorem/                   # Theorem fidelity guarantees verification
```

---

## 📜 Dependencies

```bash
pip install -r requirements.txt
```

---

LUCID Research Team - 2026