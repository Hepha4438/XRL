import torch
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train_sae_logic import SAELogicAgentV3, SAELogicConfig 

def run_theorem2_analysis(model_path, data_path, output_dir, tau=0.5):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading LUCID model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = SAELogicConfig(**checkpoint['config'])
    
    model = SAELogicAgentV3(config, device=device)
    model.load_state_dict(checkpoint['model_state'])
    # Phục hồi stats chuẩn hóa từ checkpoint
    model.set_normalization(checkpoint['feature_mean'].to(device), checkpoint['feature_std'].to(device))
    if 'z_mean' in checkpoint:
        model.z_mean.copy_(checkpoint['z_mean'].to(device))
        model.z_std.copy_(checkpoint['z_std'].to(device))
    
    model.to(device) 
    model.eval()

    print(f"Loading data: {data_path}")
    data = torch.load(data_path, map_location=device, weights_only=False)
    features = data['features'].to(device)
    
    alpha_j = model.bottleneck.get_sharpness().detach() 
    beta_j = model.bottleneck.beta.detach()
    p_probs, n_probs = model.logic_layer._get_selection_probs()
    
    n_actions = model.config.n_actions
    n_clauses_per_action = model.config.n_clauses_per_action
    
    # -------------------------------------------------------------------------
    # TRÍCH XUẤT LUẬT CỨNG (Tương đương với việc in ra Text)
    # -------------------------------------------------------------------------
    P_mask = (p_probs.detach() > tau) # Nhóm Literal Dương (Positive)
    N_mask = (n_probs.detach() > tau) # Nhóm Literal Âm (Negative)
    active_mask = P_mask | N_mask 
    global_active_concepts = active_mask.any(dim=0)
    
    cb = model.logic_layer.clause_weight.detach() # Lấy Bias gốc
    clause_sigmoid_weights = torch.sigmoid(cb)    # Tính sẵn Sigmoid(Bias) cho luật True
    # -------------------------------------------------------------------------
    
    results = []
    fragile_eps_accum = torch.zeros(model.config.hidden_dim, device=device)
    
    print(f"Processing Theorem 2 calculations (Tau={tau})...")
    
    for x in tqdm(features):
        with torch.no_grad():
            x = x.unsqueeze(0)
            
            # --- QUAN TRỌNG: CHUẨN HÓA DỮ LIỆU ĐẦU VÀO ---
            x_norm = model.normalize_input(x)
            
            # --- T2-1: TÍNH HARD SCORE (Hard Concept + Hard Text Rule) ---
            z_sparse, _ = model.sae.encode(x_norm) # Đã đổi x thành x_norm
            z_prime = (z_sparse - model.z_mean) / model.z_std
            c_hard = (z_prime > beta_j) # Tensor Boolean (True/False)
            has_active_lits = P_mask.any(dim=1) | N_mask.any(dim=1)
            
            # Kiểm tra luật bằng Logic Boolean khắt khe
            p_violation = P_mask & (~c_hard)
            n_violation = N_mask & c_hard
            clause_violation = p_violation.any(dim=1) | n_violation.any(dim=1)
            clause_is_true = (~clause_violation) & has_active_lits
            
            # Nếu mệnh đề True (~violation) và không rỗng, điểm là Sigmoid(bias). Nếu False, điểm là 0.
            clause_scores = clause_sigmoid_weights * clause_is_true.float()
            
            # Cộng dồn điểm cho từng hành động
            clause_scores = clause_scores.view(n_actions, n_clauses_per_action)
            s_hard = clause_scores.sum(dim=1) # Shape: [n_actions]
            
            top_scores, top_indices = torch.topk(s_hard, k=2)
            gamma = top_scores[0] - top_scores[1]

            # --- T2-2 & T2-3: Tính Perturbation Delta(x) (Concept binarization error) ---
            delta_j = torch.abs(z_prime - beta_j).squeeze(0)
            eps_j = 1.0 / (1.0 + torch.exp(alpha_j * delta_j))
            
            log_one_minus_eps = torch.log(1.0 - eps_j + 1e-10)
            clause_log_prod = torch.sum(active_mask * log_one_minus_eps, dim=1)
            clause_deltas = 1.0 - torch.exp(clause_log_prod) 
            global_delta = torch.sum(clause_deltas).item()

            # --- T2-5: So sánh với mạng Soft (Soft Concept + Soft Logic) ---
            s_soft = model(x_norm).squeeze(0) # Đã đổi x thành x_norm
            c_soft = model.bottleneck(z_prime).squeeze(0)
            
            # So sánh Action
            agreement = (s_soft.argmax().item() == s_hard.argmax().item())
            
            results.append({
                'gamma': gamma.item(),
                'two_delta': 2 * global_delta,
                'agreement': agreement,
                'c_soft_samples': c_soft.cpu().numpy()
            })
            
            if gamma.item() <= 2 * global_delta:
                fragile_eps_accum += (eps_j * global_active_concepts)

    # =========================================================================
    # THỐNG KÊ KẾT QUẢ THEO CHUẨN LATEX
    # =========================================================================
    gamma_arr = np.array([r['gamma'] for r in results])
    two_delta_arr = np.array([r['two_delta'] for r in results]) 
    agree_arr = np.array([r['agreement'] for r in results])
    
    covered_mask = gamma_arr > two_delta_arr
    uncovered_mask = ~covered_mask
    fidelity_coverage = np.mean(covered_mask) * 100
    
    print(f"\n" + "="*60)
    print(f"STEP T2-4: DISTRIBUTIONS (Mean, Median, Percentiles)")
    print(f"="*60)
    print(f"Gamma (Margin)     | Mean: {np.mean(gamma_arr):.3f} | Median: {np.median(gamma_arr):.3f} | 5th: {np.percentile(gamma_arr, 5):.3f} | 95th: {np.percentile(gamma_arr, 95):.3f}")
    print(f"2*Delta (Bound)    | Mean: {np.mean(two_delta_arr):.3f} | Median: {np.median(two_delta_arr):.3f} | 5th: {np.percentile(two_delta_arr, 5):.3f} | 95th: {np.percentile(two_delta_arr, 95):.3f}")

    print(f"\n" + "="*60)
    print(f"STEP T2-5: EMPIRICAL VALIDATION OF THEOREM 2")
    print(f"="*60)
    covered_agreement = np.mean(agree_arr[covered_mask]) * 100 if np.any(covered_mask) else 0.0
    uncovered_disagreement = np.mean(~agree_arr[uncovered_mask]) * 100 if np.any(uncovered_mask) else 0.0
    overall_agreement = np.mean(agree_arr) * 100
    
    print(f"Overall Soft vs Hard Agreement: {overall_agreement:.2f}%")
    print(f"Fidelity Coverage:              {fidelity_coverage:.2f}% (Target: ≥ 95%)")
    print(f"Covered Agreement (Invariant):  {covered_agreement:.2f}%")
    if covered_agreement < 100.0:
        print("  -> NOTE: Theorem 2 bound (Delta) only accounts for Concept Binarization.")
        print("     The remaining gap is the exact logic-thresholding error (Neuro-Symbolic Gap).")
    print(f"Uncovered Disagreement Rate:    {uncovered_disagreement:.2f}%")

    print(f"\n" + "="*60)
    print(f"STEP T2-6: FRAGILE ZONE DIAGNOSTICS")
    print(f"="*60)
    if np.any(uncovered_mask):
        print(f"In the fragile zone (Uncovered states):")
        print(f"  - Average Gamma:   {np.mean(gamma_arr[uncovered_mask]):.3f}")
        print(f"  - Average 2*Delta: {np.mean(two_delta_arr[uncovered_mask]):.3f}")
        
        top_fragile_concepts = torch.topk(fragile_eps_accum, k=5)
        print(f"  - Top concepts contributing to Delta (Needs better binarization):")
        for idx, val in zip(top_fragile_concepts.indices, top_fragile_concepts.values):
            if val > 0: print(f"      * Concept f_{idx.item()}: accumulated error = {val.item():.2f}")
    else:
        print("No fragile states detected! Perfect binarization.")
    print(f"="*60)

    save_path = os.path.join(output_dir, 'theorem2_analytics.pt')
    torch.save(results, save_path)
    print(f"\nSaved analytics to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../../sae_logic_v3_outputs/sae_logic_v3_model.pt")
    parser.add_argument("--data_path", type=str, default="../../held_out_evaluation_data.pt")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()
    
    run_theorem2_analysis(args.model_path, args.data_path, args.output_dir, args.tau)