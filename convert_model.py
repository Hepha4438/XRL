#!/usr/bin/env python3
import os
import zipfile
import tempfile
import torch
from stable_baselines3 import PPO

# Import hàm tạo môi trường của chính bạn
from test_pixel_cartpole import make_vec_env

def main():
    print("1. Khởi tạo một mô hình PPO trống (để lấy cấu trúc mạng lưới)...")
    env = make_vec_env("PixelCartPole-v0", render_mode=None, seed=42)
    # Khởi tạo mô hình mặc định (chưa có não)
    model = PPO("CnnPolicy", env, device="cpu")

    print("2. Giải nén file zip cũ để lấy trộm 'bộ não' (PyTorch weights)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Tự mở file zip mà không dùng thư viện load của Stable Baselines
        with zipfile.ZipFile("ppo_pixel_cartpole.zip", 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # Load trực tiếp file policy.pth vào mạng Pytorch (Bỏ qua hoàn toàn mảng NumPy lỗi!)
        policy_path = os.path.join(tmp_dir, "policy.pth")
        model.policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
        
    print("3. Đóng gói và lưu lại thành file mới dưới chuẩn môi trường hiện tại...")
    # Lệnh save này sẽ tự động dùng NumPy 1.26.4 của bạn để đóng gói
    model.save("ppo_cartpole_converted.zip")
    
    print("✅ HOÀN TẤT! File 'ppo_cartpole_converted.zip' đã sẵn sàng.")

if __name__ == "__main__":
    main()