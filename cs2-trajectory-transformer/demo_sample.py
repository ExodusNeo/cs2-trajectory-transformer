"""
Demo Verification Script for CS2 Trajectory Transformer Project.
Generates synthetic 128-tick telemetry, computes micro-kinematic features, 
and runs a forward pass through the PyTorch ST-Trans model.
"""

import sys
import os

# Add local src/ directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import pandas as pd
import torch

from features.kinematics import compute_kinematic_features
from models.st_transformer import STTrajectoryTransformer


def main():
    print("=" * 65)
    print("CS2 TRAJECTORY TRANSFORMER // LOCAL VERIFICATION PIPELINE")
    print("=" * 65)
    
    # 1. Generate Synthetic 128-tick Trajectory Data (10 seconds @ 128Hz = 1280 ticks)
    num_ticks = 1280
    t = np.linspace(0, 10, num_ticks)
    
    # Simulated mouse view trajectory with 10 Hz physiological hand tremor
    tremor = 0.04 * np.sin(2 * np.pi * 10.0 * t)
    yaw = 175 + 15 * np.sin(2 * np.pi * 0.4 * t) + tremor + np.random.normal(0, 0.02, num_ticks)
    pitch = 0 + 8 * np.cos(2 * np.pi * 0.25 * t) + tremor + np.random.normal(0, 0.02, num_ticks)
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
    
    print(f"[1] Generated 128-tick match telemetry sequence ({num_ticks} frames).")
    
    # 2. Compute Biomechanical Kinematic & Tremor Features
    df = compute_kinematic_features(df, tick_rate=128.0, extract_tremor=True)
    
    print("[2] Biomechanical & Tremor Feature Extraction complete:")
    summary_cols = ['angular_velocity', 'angular_jerk', 'trajectory_curvature', 'curvature_entropy', 'tremor_power_8_12hz']
    print(df[summary_cols].head())
    
    # 3. Prepare Sequence Input Tensor for PyTorch Transformer (8 Feature Dimensions)
    feature_cols = [
        'yaw', 'pitch', 'angular_velocity', 'angular_accel', 
        'angular_jerk', 'trajectory_curvature', 'curvature_entropy', 'tremor_power_8_12hz'
    ]
    features_array = df[feature_cols].values
    
    # Standardize features
    features_normalized = (features_array - np.mean(features_array, axis=0)) / (np.std(features_array, axis=0) + 1e-6)
    
    # Reshape to (batch_size=1, seq_len=1280, feature_dim=8)
    input_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
    
    print(f"\n[3] Model Input Tensor Shape: {input_tensor.shape} (batch, ticks, features)")
    
    # 4. Instantiate & Run PyTorch ST-Trans Model
    model = STTrajectoryTransformer(feature_dim=len(feature_cols), d_model=64, nhead=4, num_layers=2)
    model.eval()
    
    with torch.no_grad():
        aimbot_prob, smurf_embedding, predicted_elo = model(input_tensor)
        
    print("\n" + "=" * 65)
    print("MODEL INFERENCE RESULTS:")
    print("=" * 65)
    print(f"  > Aimbot Probability (Organic vs Algorithmic): {aimbot_prob.item()*100:.2f}%")
    print(f"  > Smurf Latent Embedding Shape (InfoNCE): {smurf_embedding.shape} (L2 Norm: {torch.norm(smurf_embedding).item():.2f})")
    print(f"  > Predicted Player Skill ELO: {predicted_elo.item() * 2000:.0f} ELO")
    print("=" * 65)
    print("Verification Successful! Biomechanical extraction & ST-Trans are synchronized.")


if __name__ == "__main__":
    main()
