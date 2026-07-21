"""
Demo Verification Script for CS2 Trajectory Transformer Project.
Generates synthetic 128-tick telemetry, computes kinematic features, 
and runs a forward pass through the PyTorch ST-Trans model.
"""

import sys
import os

# Add local src/ directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import pandas as pd
import torch

from features.kinematics import compute_kinematic_features, calculate_windowed_entropy
from models.st_transformer import STTrajectoryTransformer


def main():
    print("=" * 60)
    print("CS2 TRAJECTORY TRANSFORMER // LOCAL VERIFICATION SCRIPT")
    print("=" * 60)
    
    # 1. Generate Synthetic 128-tick Trajectory Data (10 seconds @ 128Hz = 1280 ticks)
    num_ticks = 1280
    t = np.linspace(0, 10, num_ticks)
    
    # Human-like noisy sinusoidal view trajectory
    yaw = 180 + 25 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.05, num_ticks)
    pitch = 0 + 10 * np.cos(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.05, num_ticks)
    pos_x = 1000 + 50 * t
    pos_y = 500 + 10 * np.sin(t)
    pos_z = 64.0
    
    df = pd.DataFrame({
        'tick': np.arange(num_ticks),
        'yaw': yaw,
        'pitch': pitch,
        'x': pos_x,
        'y': pos_y,
        'z': pos_z
    })
    
    print(f"[1] Simulated 128-tick match telemetry sequence ({num_ticks} frames).")
    
    # 2. Compute Biomechanical Kinematic Features
    df = compute_kinematic_features(df, tick_rate=128.0)
    df['entropy'] = calculate_windowed_entropy(df['angular_jerk'].values, window_size=32)
    
    print("[2] Biomechanical Feature Extraction complete:")
    print(df[['angular_velocity', 'angular_jerk', 'trajectory_curvature', 'entropy']].head())
    
    # 3. Prepare Sequence Input Tensor for PyTorch Transformer
    feature_cols = ['yaw', 'pitch', 'angular_velocity', 'angular_accel', 'angular_jerk', 'entropy']
    features_array = df[feature_cols].values
    
    # Normalize features
    features_normalized = (features_array - features_array.mean(axis=0)) / (features_array.std(axis=0) + 1e-6)
    
    # Reshape to (batch_size=1, seq_len=1280, feature_dim=6)
    input_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
    
    print(f"\n[3] Model Input Tensor Shape: {input_tensor.shape} (batch, ticks, features)")
    
    # 4. Instantiate & Run PyTorch ST-Trans Model
    model = STTrajectoryTransformer(feature_dim=6, d_model=64, nhead=4, num_layers=2)
    model.eval()
    
    with torch.no_grad():
        aimbot_prob, predicted_elo = model(input_tensor)
        
    print("\n" + "=" * 60)
    print("MODEL INFERENCE RESULTS:")
    print("=" * 60)
    print(f"  > Aimbot Probability (Organic vs Algorithmic): {aimbot_prob.item()*100:.2f}%")
    print(f"  > Predicted Player Skill ELO: {predicted_elo.item() * 2000:.0f} ELO")
    print("=" * 60)
    print("Verification Successful! Your project structure is ready for thesis implementation.")

if __name__ == "__main__":
    main()
