"""
High-Fidelity Synthetic CS2 Trajectory Generator for Benchmark Experiments.
Simulates biological neuromuscular motor control vs. algorithmic aimbot signatures:
1. Human Skill Tiers (Silver, Gold Nova, Faceit Level 10 Pro) with 8-12 Hz tremor.
2. Cheater Types (Hard Snap Aimbot, Smooth Aim Assist, Silent Micro-Correction).
Outputs structured 8D Parquet trajectory segments into data/processed_parquet/.
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from features.kinematics import compute_kinematic_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def generate_human_trajectory(
    n_ticks: int = 256, 
    elo: float = 2000.0, 
    tick_rate: float = 128.0
) -> pd.DataFrame:
    """Simulates human aiming with Minimum Jerk planning and 8-12 Hz biological tremor."""
    t = np.arange(n_ticks) / tick_rate
    
    # Skill-dependent parameters
    skill_factor = np.clip(elo / 2500.0, 0.2, 1.2)
    tremor_amp = np.random.uniform(0.02, 0.05)  # Natural hand tremor
    noise_amp = 0.05 / (skill_factor + 0.2)
    
    # 8-12 Hz physiological tremor
    tremor_freq = np.random.uniform(9.0, 11.5)
    tremor_yaw = tremor_amp * np.sin(2 * np.pi * tremor_freq * t + np.random.uniform(0, np.pi))
    tremor_pitch = tremor_amp * np.cos(2 * np.pi * tremor_freq * t + np.random.uniform(0, np.pi))
    
    # Minimum jerk aiming path to a target
    # Target acquisition trajectory using 5th order polynomial (Flash & Hogan)
    t_norm = np.clip(t / (0.3 / skill_factor + 0.1), 0.0, 1.0)
    target_yaw = np.random.uniform(-45, 45)
    target_pitch = np.random.uniform(-20, 20)
    
    # Minimum jerk polynomial: 10*t^3 - 15*t^4 + 6*t^5
    poly = 10 * (t_norm**3) - 15 * (t_norm**4) + 6 * (t_norm**5)
    
    yaw = 180.0 + poly * target_yaw + tremor_yaw + np.random.normal(0, noise_amp, n_ticks)
    pitch = 0.0 + poly * target_pitch + tremor_pitch + np.random.normal(0, noise_amp, n_ticks)
    
    # Position
    x = np.linspace(500, 700, n_ticks)
    y = np.linspace(200, 300, n_ticks)
    z = np.full(n_ticks, 64.0)
    
    df = pd.DataFrame({'tick': np.arange(n_ticks), 'yaw': yaw, 'pitch': pitch, 'X': x, 'Y': y, 'Z': z})
    return df


def generate_cheater_trajectory(
    n_ticks: int = 256, 
    cheat_type: str = 'snap', 
    tick_rate: float = 128.0
) -> pd.DataFrame:
    """Simulates algorithmic aimbot behaviors (Zero Tremor, Linear Interpolation, Instant Snaps)."""
    t = np.arange(n_ticks) / tick_rate
    
    target_yaw = np.random.uniform(-45, 45)
    target_pitch = np.random.uniform(-20, 20)
    
    if cheat_type == 'snap':
        # Hard snap: instantaneous jump in 1-2 ticks, then zero movement
        yaw = np.full(n_ticks, 180.0)
        pitch = np.full(n_ticks, 0.0)
        snap_tick = np.random.randint(20, 50)
        yaw[snap_tick:] = 180.0 + target_yaw
        pitch[snap_tick:] = target_pitch
        # Synthetic cheat noise (quantized / algorithmic)
        yaw += np.random.normal(0, 0.002, n_ticks)
        pitch += np.random.normal(0, 0.002, n_ticks)
        
    elif cheat_type == 'smooth':
        # Smooth aimbot: perfectly linear or constant angular speed without biological tremor
        yaw = np.full(n_ticks, 180.0)
        pitch = np.full(n_ticks, 0.0)
        start_t = np.random.randint(20, 40)
        dur = np.random.randint(15, 30)
        for i in range(dur):
            progress = (i + 1) / dur
            yaw[start_t + i:] = 180.0 + progress * target_yaw
            pitch[start_t + i:] = progress * target_pitch
            
    else:  # silent / micro
        yaw = 180.0 + 5.0 * np.sin(2 * np.pi * 0.5 * t)
        pitch = 0.0 + 3.0 * np.cos(2 * np.pi * 0.3 * t)
        # Micro correction spikes during shot ticks
        shot_ticks = [60, 120, 180]
        for st in shot_ticks:
            if st < n_ticks:
                yaw[st:st+3] += target_yaw * 0.5
                pitch[st:st+3] += target_pitch * 0.5
                
    x = np.linspace(500, 700, n_ticks)
    y = np.linspace(200, 300, n_ticks)
    z = np.full(n_ticks, 64.0)
    
    df = pd.DataFrame({'tick': np.arange(n_ticks), 'yaw': yaw, 'pitch': pitch, 'X': x, 'Y': y, 'Z': z})
    return df


def create_synthetic_dataset(output_dir: str = "data/processed_parquet", num_samples: int = 600):
    """Generates partitioned benchmark dataset of clean players and cheaters."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 20 distinct players: 10 clean (varying ELOs), 10 cheaters
    clean_players = [
        (76561198000000010 + i, np.random.choice([600, 1400, 2600])) 
        for i in range(10)
    ]
    cheater_players = [
        (76561198000000050 + i, np.random.choice(['snap', 'smooth', 'silent'])) 
        for i in range(10)
    ]
    
    samples_per_player = num_samples // (len(clean_players) + len(cheater_players))
    total_saved = 0
    
    # 1. Clean Players
    for p_id, elo in clean_players:
        for seg in range(samples_per_player):
            n_ticks = np.random.randint(128, 256)
            raw_df = generate_human_trajectory(n_ticks=n_ticks, elo=elo)
            feat_df = compute_kinematic_features(raw_df, tick_rate=128.0, extract_tremor=True)
            
            feat_df['match_id'] = f"match_c{p_id % 5}"
            feat_df['steamid'] = p_id
            feat_df['segment_id'] = seg
            feat_df['is_aimbot'] = 0
            feat_df['player_elo'] = float(elo)
            
            out_file = os.path.join(output_dir, f"match_c{p_id % 5}_p{p_id}_seg{seg}.parquet")
            feat_df.to_parquet(out_file, index=False)
            total_saved += 1
            
    # 2. Cheater Players
    for p_id, cheat_type in cheater_players:
        for seg in range(samples_per_player):
            n_ticks = np.random.randint(128, 256)
            raw_df = generate_cheater_trajectory(n_ticks=n_ticks, cheat_type=cheat_type)
            feat_df = compute_kinematic_features(raw_df, tick_rate=128.0, extract_tremor=True)
            
            feat_df['match_id'] = f"match_x{p_id % 5}"
            feat_df['steamid'] = p_id
            feat_df['segment_id'] = seg
            feat_df['is_aimbot'] = 1
            feat_df['player_elo'] = 2400.0  # Cheater spoofed rank
            
            out_file = os.path.join(output_dir, f"match_x{p_id % 5}_p{p_id}_seg{seg}.parquet")
            feat_df.to_parquet(out_file, index=False)
            total_saved += 1
            
    logging.info(f"Generated {total_saved} synthetic benchmark trajectory segments in {output_dir}")
    return total_saved


if __name__ == "__main__":
    create_synthetic_dataset()
