"""
Feature Ablation Study Pipeline for Spatial-Temporal Trajectory Transformer (ST-Trans).

Evaluates the empirical contribution of each biomechanical invariant:
1. Full Model (8 Features: yaw, pitch, vel, accel, jerk, curvature, tortuosity, tremor)
2. Ablation A: w/o 8-12 Hz Tremor PSD (Tests neuromuscular micro-tremor invariant)
3. Ablation B: w/o Angular Jerk (Tests Flash & Hogan minimum-jerk optimization)
4. Ablation C: w/o Spherical Curvature & Tortuosity (Tests spherical S^2 geodesic geometry)
5. Ablation D: Kinematics Only (Tests coordinate invariance: no raw yaw/pitch)
6. Ablation E: Raw Coordinates Only (Tests baseline transformer with zero biomechanical features)

Outputs:
- reports/ablation_study_summary.csv
- reports/ablation_study.png
"""

import os
import sys

# Automatically route to nested cs2-trajectory-transformer directory if run from root
base_dir = os.path.dirname(os.path.abspath(__file__))
nested_dir = os.path.join(base_dir, 'cs2-trajectory-transformer')
if os.path.exists(nested_dir):
    sys.path.insert(0, nested_dir)
    os.chdir(nested_dir)

from ablation import main if 'main' in locals() else run_ablation_study
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Ablation Study for CS2 Trajectory Transformer")
    parser.add_argument("--data_dir", type=str, default="data/processed_parquet", help="Path to processed parquet data")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs per ablation configuration")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible partitions")
    args = parser.parse_args()

    run_ablation_study(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
