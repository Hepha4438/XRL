import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

# Thêm thư mục gốc vào path để import train_sae_logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train_sae_logic import SAELogicAgentV3, SAELogicConfig 

def run_theorem2_analysis(model_path, data_path, output_dir, tau=0.3):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model và cấu hình
    print(f"Loading LUCID model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = SAELogicConfig(**checkpoint['config'])
    
    model = SAELogicAgentV3(config, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device) 
    model.eval()

    # 2. Load dữ liệu Held-out
    print(f"Loading data: {data_path}")
    data = torch.load(data_path, map_location=device, weights_only=False)
    features = data['features'].to(device)
    
    # 3. Chuẩn bị tham số trích xuất từ logic_layer
    alpha_j = model.bottleneck.get_sharpness().detach() 
    beta_j = model.bottleneck.beta.detach()
    p_probs, n_probs = model.logic_layer._get_selection_probs()
    
    n_actions = model.logic_layer.n_actions
    n_clauses_per_action = model.logic_layer.n_clauses_per_action
    
    active_mask = (p_probs.detach() > tau) | (n_probs.detach() > tau) 
    
    results = []
    print(f"Processing Theorem 2 calculations (Tau={tau})...")
    
    for x in tqdm(features):
        with torch.no_grad():
            x = x.unsqueeze(0)
            
            # --- T2-1: Action Margin Gamma(x) ---
            z_sparse, _ = model.sae.encode(x)
            z_prime = (z_sparse - model.z_mean) / model.z_std
            c_hard = (z_prime > beta_j).float()
            
            # Điểm số từ quy tắc cứng
            s_hard_logits = model.logic_layer(c_hard).squeeze(0)
            
            top_scores, top_indices = torch.topk(s_hard_logits, k=2)
            gamma = top_scores[0] - top_scores[1]

            # --- T2-2 & T2-3: Tính Delta(x) toàn cục theo đúng Theorem 2 ---
            delta_j = torch.abs(z_prime - beta_j).squeeze(0)
            eps_j = 1.0 / (1.0 + torch.exp(alpha_j * delta_j))
            
            log_one_minus_eps = torch.log(1.0 - eps_j + 1e-10)
            clause_log_prod = torch.sum(active_mask * log_one_minus_eps, dim=1)
            clause_deltas = 1.0 - torch.exp(clause_log_prod) 
            
            # Delta(x) theo đúng định nghĩa lý thuyết: Tổng sai số của TẤT CẢ các mệnh đề
            global_delta = torch.sum(clause_deltas).item()

            # --- T2-5: Sanity Check & Binarisation samples ---
            s_soft = model(x).squeeze(0)
            c_soft = model.bottleneck(z_prime).squeeze(0)
            
            results.append({
                'gamma': gamma.item(),
                'two_delta': 2 * global_delta, # Khớp với trục Y trong LaTeX
                'agreement': (s_soft.argmax().item() == s_hard_logits.argmax().item()),
                'c_soft_samples': c_soft.cpu().numpy()
            })

    # --- T2-4: Thống kê và Báo cáo ---
    gamma_arr = np.array([r['gamma'] for r in results])
    two_delta_arr = np.array([r['two_delta'] for r in results]) # Đổi từ 'delta' thành 'two_delta'
    agree_arr = np.array([r['agreement'] for r in results])
    
    # Fidelity Coverage theo điều kiện Theorem 2: gamma > 2*delta
    covered_mask = gamma_arr > two_delta_arr
    fidelity_coverage = np.mean(covered_mask) * 100
    
    # Kiểm tra tính đúng đắn (Phải là 100%)
    if np.any(covered_mask):
        covered_agreement = np.mean(agree_arr[covered_mask]) * 100
    else:
        covered_agreement = 0.0

    print(f"\n" + "="*50)
    print(f"THEOREM 2: FIDELITY COVERAGE ANALYSIS")
    print(f"="*50)
    print(f"Total States:            {len(results)}")
    print(f"Overall Soft-Hard Agree: {np.mean(agree_arr)*100:.2f}%")
    print(f"Fidelity Coverage:       {fidelity_coverage:.2f}%")
    print(f"Agreement (Covered):     {covered_agreement:.2f}% (Invariant Check)")
    print(f"="*50)

    # Lưu kết quả cho bước Visualization (T2-6 & T2-7)
    save_path = os.path.join(output_dir, 'theorem2_analytics.pt')
    torch.save(results, save_path)
    print(f"Saved analytics to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../../sae_logic_v3_outputs/sae_logic_v3_model.pt")
    parser.add_argument("--data_path", type=str, default="../../held_out_evaluation_data.pt")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tau", type=float, default=0.3)
    args = parser.parse_args()
    
    run_theorem2_analysis(args.model_path, args.data_path, args.output_dir, args.tau)