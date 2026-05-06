import torch
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train_joint import SAELogicAgentV3, SAELogicConfig 

def run_theorem2_analysis(model_path, data_path, output_dir, tau=0.5):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading LUCID model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = SAELogicConfig(**checkpoint['config'])
    
    model = SAELogicAgentV3(config, device=device)
    model.load_state_dict(checkpoint['model_state'])
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
    ignore_probs = 1.0 - p_probs - n_probs
    
    P_mask = (p_probs.detach() > tau) 
    N_mask = (n_probs.detach() > tau) 
    active_mask = P_mask | N_mask
    global_active_concepts = active_mask.any(dim=0)
    
    n_actions = model.config.n_actions
    n_clauses_per_action = model.config.n_clauses_per_action
    
    cb = model.logic_layer.clause_weight.detach() 
    clause_sigmoid_weights = torch.sigmoid(cb)    
    
    results = []
    fragile_eps_accum = torch.zeros(model.config.hidden_dim, device=device)
    
    print(f"Processing Complete Theorem 2 Calculations (Tau={tau})...")
    
    for x in tqdm(features):
        with torch.no_grad():
            x = x.unsqueeze(0)
            x_norm = model.normalize_input(x)
            
            # --- T2-1: HARD SCORE ---
            z_sparse, _ = model.sae.encode(x_norm)
            z_prime = (z_sparse - model.z_mean) / model.z_std
            c_hard = (z_prime > beta_j) 
            has_active_lits = P_mask.any(dim=1) | N_mask.any(dim=1)
            
            p_violation = P_mask & (~c_hard)
            n_violation = N_mask & c_hard
            clause_violation = p_violation.any(dim=1) | n_violation.any(dim=1)
            clause_is_true = (~clause_violation) & has_active_lits
            
            clause_scores = clause_sigmoid_weights * clause_is_true.float()
            clause_scores = clause_scores.view(n_actions, n_clauses_per_action)
            s_hard = clause_scores.sum(dim=1)
            
            top_scores, top_indices = torch.topk(s_hard, k=2)
            gamma = top_scores[0] - top_scores[1]
            top_action = top_indices[0].item()

            # --- T2-2 & T2-3: EXACT PER-SAMPLE DELTA(x) ---
            c_soft = model.bottleneck(z_prime).squeeze(0) # [D]
            l_soft = p_probs * c_soft + n_probs * (1.0 - c_soft) + ignore_probs # [n_clauses, D]
            
            # TRUE CLAUSE ERROR
            clause_log_prod = torch.sum(torch.log(l_soft + 1e-10), dim=1) 
            true_clause_deltas = clause_sigmoid_weights - torch.sigmoid(cb + clause_log_prod)
            
            # FALSE CLAUSE ERROR (ĐÃ FIX THEO CHUẨN LEMMA 1)
            violated_mask = p_violation | n_violation
            
            # 1. Tìm literal vi phạm NHỎ NHẤT (Tightest Bound)
            # Lót vô cùng (inf) cho các literal KHÔNG vi phạm để hàm min() bỏ qua chúng
            masked_l_soft_min = torch.where(violated_mask, l_soft, torch.tensor(float('inf'), device=device))
            l_soft_star_min, _ = torch.min(masked_l_soft_min, dim=1)
            
            # Tính sai số cho False Clause bình thường
            false_clause_violated_deltas = torch.sigmoid(cb + torch.log(l_soft_star_min + 1e-10))
            
            # 2. Xử lý lỗi EMPTY CLAUSE (Luật rỗng)
            # Hard = 0, Soft = C_soft. Sai số chính là toàn bộ giá trị Soft!
            empty_clause_mask = ~has_active_lits
            empty_clause_deltas = torch.sigmoid(cb + clause_log_prod) # Đây chính là C_soft hiện tại
            
            # Kết hợp: Nếu là empty clause thì lấy empty_deltas, nếu không thì lấy violated_deltas
            false_clause_deltas = torch.where(empty_clause_mask, empty_clause_deltas, false_clause_violated_deltas)
            
            clause_deltas = torch.where(clause_is_true, true_clause_deltas, false_clause_deltas)
            
            # --- TIGHTER BOUND ---
            clause_deltas_per_action = clause_deltas.view(n_actions, n_clauses_per_action)
            delta_a = clause_deltas_per_action.sum(dim=1) 
            
            delta_a_star = delta_a[top_action].item()
            mask_other = torch.ones(n_actions, dtype=torch.bool, device=device)
            mask_other[top_action] = False
            max_delta_other = torch.max(delta_a[mask_other]).item()
            
            tight_fidelity_bound = delta_a_star + max_delta_other
            
            # --- T2-5: Compare to Soft one ---
            s_soft = model(x_norm).squeeze(0)
            agreement = (s_soft.argmax().item() == s_hard.argmax().item())
            
            results.append({
                'gamma': gamma.item(),
                'delta_bound': tight_fidelity_bound,
                'agreement': agreement,
                'c_soft_samples': c_soft.cpu().numpy()
            })
            
            if gamma.item() <= tight_fidelity_bound:
                fragile_eps_accum += ((1.0 - l_soft).max(dim=0)[0] * global_active_concepts)

    gamma_arr = np.array([r['gamma'] for r in results])
    delta_bound_arr = np.array([r['delta_bound'] for r in results]) 
    agree_arr = np.array([r['agreement'] for r in results])
    
    covered_mask = gamma_arr > delta_bound_arr
    uncovered_mask = ~covered_mask
    fidelity_coverage = np.mean(covered_mask) * 100
    
    print(f"\n" + "="*60)
    print(f"STEP T2-4: DISTRIBUTIONS (Mean, Median, Percentiles)")
    print(f"="*60)
    print(f"Gamma (Margin)     | Mean: {np.mean(gamma_arr):.3f} | Median: {np.median(gamma_arr):.3f} | 5th: {np.percentile(gamma_arr, 5):.3f} | 95th: {np.percentile(gamma_arr, 95):.3f}")
    print(f"Tight Bound (T2)   | Mean: {np.mean(delta_bound_arr):.3f} | Median: {np.median(delta_bound_arr):.3f} | 5th: {np.percentile(delta_bound_arr, 5):.3f} | 95th: {np.percentile(delta_bound_arr, 95):.3f}")

    print(f"\n" + "="*60)
    print(f"STEP T2-5: EMPIRICAL VALIDATION OF COMPLETE THEOREM 2")
    print(f"="*60)
    covered_agreement = np.mean(agree_arr[covered_mask]) * 100 if np.any(covered_mask) else 0.0
    uncovered_disagreement = np.mean(~agree_arr[uncovered_mask]) * 100 if np.any(uncovered_mask) else 0.0
    overall_agreement = np.mean(agree_arr) * 100
    
    print(f"Overall Soft vs Hard Agreement: {overall_agreement:.2f}%")
    print(f"Fidelity Coverage:              {fidelity_coverage:.2f}%")
    print(f"Covered Agreement (Invariant):  {covered_agreement:.2f}%")
    
    if covered_agreement < 100.0:
        print("  -> WARNING: Covered Agreement < 100%. This breaks the theoretical invariant!")
    else:
        print("  -> PERFECT! The combined bound (Concept + Selector Softness) successfully guarantees 100% agreement.")
        
    print(f"Uncovered Disagreement Rate:    {uncovered_disagreement:.2f}%")

    print(f"\n" + "="*60)
    print(f"STEP T2-6: FRAGILE ZONE DIAGNOSTICS")
    print(f"="*60)
    if np.any(uncovered_mask):
        print(f"In the fragile zone (Uncovered states):")
        print(f"  - Average Gamma:       {np.mean(gamma_arr[uncovered_mask]):.3f}")
        print(f"  - Average Tight Bound: {np.mean(delta_bound_arr[uncovered_mask]):.3f}")
        
        top_fragile_concepts = torch.topk(fragile_eps_accum, k=5)
        print(f"  - Top concepts contributing to Combined Delta (Needs better binarization/entropy):")
        for idx, val in zip(top_fragile_concepts.indices, top_fragile_concepts.values):
            if val > 0: print(f"      * Concept f_{idx.item()}: accumulated combined error = {val.item():.2f}")
    else:
        print("No fragile states detected! Perfect logic crystallization.")
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