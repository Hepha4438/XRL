import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_theorem2_visuals(results_path, output_dir):
    # Fix lỗi weights_only cho PyTorch 2.6+
    if not os.path.exists(results_path):
        print(f"Error: Not exist {results_path}")
        return

    print(f"Loading analytics from {results_path}...")
    results = torch.load(results_path, weights_only=False)
    
    gamma = np.array([r['gamma'] for r in results])
    y_axis_2delta = np.array([r['two_delta'] for r in results]) # Trục Y: 2*Delta
    agreement = np.array([r['agreement'] for r in results])
    all_c_soft = np.concatenate([r['c_soft_samples'] for r in results])

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP T2-6: FRAGILE ZONE ANALYSIS (SCATTER PLOT)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 7))
    idx_agree = (agreement == True)
    idx_disagree = (agreement == False)

    # Đã sửa: Chuẩn hóa marker: Green (Match), Red (Mismatch) đều dùng marker tròn 'o'
    # Các điểm đỏ (Disagreement) được đặt s lớn hơn và plotted sau để nổi bật
    plt.scatter(gamma[idx_agree], y_axis_2delta[idx_agree], 
                c='green', alpha=0.3, s=10, marker='o', label=r'$\pi^{soft} = \pi^{hard}$')
    plt.scatter(gamma[idx_disagree], y_axis_2delta[idx_disagree], 
                c='red', alpha=0.9, s=25, marker='o', label=r'$\pi^{soft} \neq \pi^{hard}$')

    # Đường chéo y = x (Fidelity Boundary)
    max_val = max(gamma.max(), y_axis_2delta.max()) if len(gamma) > 0 else 1
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.8, label='Fidelity Boundary ($y=x$)')
    
    # Vùng an toàn (Guaranteed Safe)
    x_fill = np.linspace(0, max_val, 100)
    plt.fill_between(x_fill, 0, x_fill, color='green', alpha=0.1, label='Guaranteed safe (Theorem 2)')

    plt.xlabel(r'Action Margin $\gamma(x)$', fontsize=13)
    plt.ylabel(r'Perturbation Bound $2\Delta(x)$', fontsize=13)
    plt.title('Figure 2: Fragile Zone Analysis (Theorem 2 Validation)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.2)
    
    # Đã sửa: Lưu định dạng PNG đúng như yêu cầu
    plt.savefig(os.path.join(output_dir, 't2_6_scatter_2delta.png'), bbox_inches='tight', dpi=300)
    print(f"Saved Scatter Plot to: {os.path.join(output_dir, 't2_6_scatter_2delta.png')}")

    # ---------------------------------------------------------
    # STEP T2-7: BINARISATION QUALITY (HISTOGRAM)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    
    # Vẽ Histogram của concept activations (PNG là mặc định phù hợp cho supplementary figure)
    plt.hist(all_c_soft, bins=100, color='purple', alpha=0.7, log=True)
    
    # Tính toán các chỉ số "Clean" vs "Ambiguous" theo LaTeX
    clean_mask = (all_c_soft <= 0.05) | (all_c_soft >= 0.95)
    ambiguous_mask = (all_c_soft > 0.05) & (all_c_soft < 0.95)
    
    clean_frac = np.mean(clean_mask) * 100
    ambig_frac = np.mean(ambiguous_mask) * 100

    plt.axvspan(0, 0.05, color='green', alpha=0.1)
    plt.axvspan(0.95, 1, color='green', alpha=0.1)
    
    plt.xlabel(r'Concept Activation Value $c_j^{soft}$', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.title('Step T2-7: Binarisation Quality', fontsize=14)
    
    # Thêm text report vào biểu đồ
    plt.text(0.5, plt.ylim()[1]*0.1, 
             f"Clean [0, 0.05]∪[0.95, 1.0]: {clean_frac:.2f}%\nAmbiguous (0.05, 0.95): {ambig_frac:.2f}%", 
             bbox=dict(facecolor='white', alpha=0.8), ha='center')

    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, 't2_7_binarization_hist.png'), bbox_inches='tight')
    print(f"Saved Histogram to: {os.path.join(output_dir, 't2_7_binarization_hist.png')}")

    # ---------------------------------------------------------
    # BÁO CÁO KẾT QUẢ
    # ---------------------------------------------------------
    # Đã sửa lỗi NameError: Đổi delta_bound thành y_axis_2delta
    fidelity_coverage = np.mean(gamma > y_axis_2delta) * 100
    overall_agreement = np.mean(agreement) * 100

    print(f"\n--- NeurIPS Report Metrics ---")
    print(f"Rigorous Fidelity Coverage (gamma > 2*Delta): {fidelity_coverage:.2f}%")
    print(f"Empirical Soft-Hard Agreement:            {overall_agreement:.2f}%")
    print(f"Binarization Cleanliness:                 {clean_frac:.2f}%")
    print(f"Binarization Ambiguity:                   {ambig_frac:.2f}%")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Chú ý: verify_fidelity.py phải được chạy trước để tạo file này
    RESULTS_FILE = os.path.join(base_dir, "results", "theorem2_analytics.pt")
    OUTPUT_FOLDER = os.path.join(base_dir, "results", "figures")
    plot_theorem2_visuals(RESULTS_FILE, OUTPUT_FOLDER)