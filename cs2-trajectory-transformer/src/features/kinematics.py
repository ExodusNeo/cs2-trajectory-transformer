"""
Kinematic Feature Extraction Module for CS2 / FPS Trajectory Analytics.
Computes Angular Velocity, Angular Jerk, and Trajectory Curvature Entropy.
"""

import numpy as np
import pandas as pd


def compute_kinematic_features(df: pd.DataFrame, tick_rate: float = 128.0) -> pd.DataFrame:
    """
    Transforms raw pitch/yaw view angles into biomechanical motor metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'yaw' (degrees) and 'pitch' (degrees) columns.
    tick_rate : float
        Frequency of telemetry sampling (default 128.0 Hz).
        
    Returns:
    --------
    pd.DataFrame with added kinematic columns:
        - angular_velocity (rad/s)
        - angular_accel (rad/s^2)
        - angular_jerk (rad/s^3)
        - curvature_entropy
    """
    dt = 1.0 / tick_rate
    
    # Handle angle wrapping
    yaw_rad = np.radians(df['yaw'].values)
    pitch_rad = np.radians(df['pitch'].values)
    
    d_yaw = np.diff(yaw_rad, prepend=yaw_rad[0])
    d_pitch = np.diff(pitch_rad, prepend=pitch_rad[0])
    
    # Angular Velocity (\omega_t)
    angular_velocity = np.sqrt(d_yaw**2 + d_pitch**2) / dt
    
    # Angular Acceleration & Jerk (j_t)
    angular_accel = np.diff(angular_velocity, prepend=angular_velocity[0]) / dt
    angular_jerk = np.diff(angular_accel, prepend=angular_accel[0]) / dt
    
    # Trajectory Curvature Proxy
    curvature = np.abs(d_pitch * np.diff(d_yaw, prepend=d_yaw[0]) - d_yaw * np.diff(d_pitch, prepend=d_pitch[0])) / (angular_velocity + 1e-6)**1.5
    
    df['angular_velocity'] = angular_velocity
    df['angular_accel'] = angular_accel
    df['angular_jerk'] = angular_jerk
    df['trajectory_curvature'] = curvature
    
    return df


def calculate_windowed_entropy(series: np.ndarray, window_size: int = 32, num_bins: int = 10) -> np.ndarray:
    """
    Calculates sliding-window Shannon Entropy over a trajectory metric.
    """
    entropy_list = []
    half_w = window_size // 2
    n = len(series)
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w)
        sub_window = series[start:end]
        
        hist, _ = np.histogram(sub_window, bins=num_bins, density=True)
        hist = hist[hist > 0]
        
        if len(hist) > 0:
            prob = hist / np.sum(hist)
            ent = -np.sum(prob * np.log2(prob))
        else:
            ent = 0.0
            
        entropy_list.append(ent)
        
    return np.array(entropy_list)
