"""
Biomechanical & Kinematic Feature Extraction Module for CS2 / FPS Replay Telemetry.

Extracts neuromuscular motor control metrics:
1. Shortest-path Euler Angle Differences (Wrapped in [-pi, pi])
2. Great-Circle Spherical Angular Velocity (rad/s)
3. Angular Acceleration (rad/s^2) and Minimum-Jerk Metric (rad/s^3)
4. Spherical Trajectory Curvature (Geodesic space)
5. 8-12 Hz Physiological Hand Tremor Relative Band Power (Welch PSD / FFT)
6. Sliding Window Trajectory Curvature & Jerk Shannon Entropy
"""

import numpy as np
import pandas as pd
from scipy import signal


def wrap_angle_rad(angles: np.ndarray) -> np.ndarray:
    """
    Wraps angular differences to the interval [-pi, pi].
    Eliminates artificial discontinuities across coordinate wrap boundaries.
    """
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def compute_spherical_angular_velocity(
    pitch_rad: np.ndarray, 
    yaw_rad: np.ndarray, 
    dt: float = 1.0 / 64.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes true great-circle angular velocity on the unit sphere view model.
    Uses differential spherical metric: ds^2 = d_pitch^2 + cos^2(pitch) * d_yaw^2
    """
    d_yaw_raw = np.diff(yaw_rad, prepend=yaw_rad[0])
    d_pitch_raw = np.diff(pitch_rad, prepend=pitch_rad[0])
    
    # Wrap angular differences to [-pi, pi]
    d_yaw = wrap_angle_rad(d_yaw_raw)
    d_pitch = wrap_angle_rad(d_pitch_raw)
    
    # Geodesic arc length differential on unit sphere
    # Note: CS2 pitch theta in [-pi/2, pi/2], cos(pitch) scales yaw displacement
    cos_pitch = np.cos(pitch_rad)
    arc_diff = np.sqrt(d_pitch**2 + (cos_pitch * d_yaw)**2)
    
    angular_velocity = arc_diff / dt
    return angular_velocity, d_pitch, d_yaw


def compute_tremor_band_power(
    signal_1d: np.ndarray, 
    sampling_rate: float = 64.0, 
    window_size: int = 64, 
    step_size: int = 1,
    tremor_low: float = 8.0, 
    tremor_high: float = 12.0
) -> np.ndarray:
    """
    Computes sliding-window Relative Band Power in the 8-12 Hz physiological tremor frequency band.
    
    Parameters:
    -----------
    signal_1d : np.ndarray
        1D time series (e.g. angular velocity or jerk).
    sampling_rate : float
        Telemetry sampling rate (CS2 native sub-tick simulation is 64.0 Hz).
    window_size : int
        Sliding window sample length (default 64 ticks = 1.0s at 64Hz, providing 1.0 Hz bin resolution).
    tremor_low : float
        Lower bound of human tremor frequency (8.0 Hz).
    tremor_high : float
        Upper bound of human tremor frequency (12.0 Hz).
        
    Returns:
    --------
    np.ndarray of relative tremor power (0.0 to 1.0) aligned with signal length.
    """
    n = len(signal_1d)
    tremor_power = np.zeros(n, dtype=np.float32)
    half_w = window_size // 2
    
    # Thesis Reference: Chapter 3, Equation (10) — Physiological Tremor Band Power (TBP)
    # TBP = (sum_{f=8}^{12} |X(f)|^2) / (sum_{f=1}^{30} |X(f)|^2 + eps)
    # Relative to active voluntary motor bandwidth (1.0 to 30.0 Hz). Frequencies > 30Hz are sensor noise.
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)
    tremor_mask = (freqs >= tremor_low) & (freqs <= tremor_high)
    total_mask = (freqs >= 1.0) & (freqs <= 30.0)  # Active voluntary motor bandwidth (1.0 - 30.0 Hz)
    
    hann = np.hanning(window_size)
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, start + window_size)
        sub = signal_1d[start:end]
        
        if len(sub) < window_size:
            # Pad if near edges
            sub = np.pad(sub, (0, window_size - len(sub)), mode='edge')
            
        sub_detrended = sub - np.mean(sub)
        windowed = sub_detrended * hann
        
        # FFT power spectrum
        fft_vals = np.abs(np.fft.rfft(windowed)) ** 2
        
        total_p = np.sum(fft_vals[total_mask])
        if total_p < 1e-4:
            # Static angle holding or negligible motion: 0.0 (prevents noise inflation)
            tremor_power[i] = 0.0
        else:
            tremor_p = np.sum(fft_vals[tremor_mask])
            tremor_power[i] = float(tremor_p / (total_p + 1e-9))
        
    return tremor_power


def compute_spherical_curvature(
    pitch_rad: np.ndarray, 
    yaw_rad: np.ndarray, 
    dt: float = 1.0 / 64.0, 
    eps: float = 1e-6
) -> np.ndarray:
    """
    Computes intrinsic geodesic curvature kappa_g of the 3D unit sight vector v(t) on unit sphere S^2.
    Thesis Reference: Chapter 3, Equation (8) — Spherical Geodesic Trajectory Curvature
    
    Formula:
        kappa_g(t) = |v(t) . (v'(t) x v''(t))| / (||v'(t)||^3 + eps)
        
    Geometric Invariant on S^2:
    - For any great-circle arc (shortest-distance path on sphere): kappa_g = 0.0 identically.
    - For non-planar / algorithmic Bézier curves deviating from geodesics: kappa_g > 0.
    """
    # 3D sight vector on unit sphere
    vx = np.cos(pitch_rad) * np.cos(yaw_rad)
    vy = np.cos(pitch_rad) * np.sin(yaw_rad)
    vz = np.sin(pitch_rad)
    v = np.stack([vx, vy, vz], axis=-1)  # shape (N, 3)
    
    # 1st and 2nd derivatives
    v_prime = np.gradient(v, dt, axis=0)
    v_double_prime = np.gradient(v_prime, dt, axis=0)
    
    # Cross product (v' x v'')
    cross = np.cross(v_prime, v_double_prime)
    
    # Velocity norm
    speed = np.linalg.norm(v_prime, axis=-1)
    
    # Intrinsic Geodesic Curvature: scalar triple product |v . (v' x v'')|
    scalar_triple = np.abs(np.sum(v * cross, axis=-1))
    
    # Zero guard during static gaze or near-zero speed
    static_mask = speed < 1e-3
    geodesic_curvature = np.zeros_like(speed)
    geodesic_curvature[~static_mask] = scalar_triple[~static_mask] / (speed[~static_mask]**3 + eps)
    
    # Clip extreme outlier artifacts
    geodesic_curvature = np.clip(geodesic_curvature, 0.0, 50.0)
    return geodesic_curvature


def calculate_windowed_entropy(series: np.ndarray, window_size: int = 32, num_bins: int = 10) -> np.ndarray:
    """
    Calculates sliding-window Shannon Entropy over a trajectory metric.
    """
    entropy_list = np.zeros(len(series), dtype=np.float32)
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
            
        entropy_list[i] = ent
        
    return entropy_list


def compute_kinematic_features(
    df: pd.DataFrame, 
    tick_rate: float = 64.0, 
    extract_tremor: bool = True
) -> pd.DataFrame:
    """
    Transforms raw pitch/yaw view angles into biomechanical motor metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'yaw' (degrees) and 'pitch' (degrees) columns.
    tick_rate : float
        Frequency of telemetry sampling (default 64.0 Hz).
    extract_tremor : bool
        Whether to calculate the 8-12 Hz tremor relative power feature.
        
    Returns:
    --------
    pd.DataFrame with added kinematic columns:
        - angular_velocity (rad/s)
        - angular_accel (rad/s^2)
        - angular_jerk (rad/s^3, speed-derived scalar angular jerk d^2 omega / dt^2)
        - trajectory_curvature (geodesic unit sphere curvature)
        - curvature_entropy
        - tremor_power_8_12hz
    """
    dt = 1.0 / tick_rate
    
    # Convert degrees to radians
    yaw_rad = np.radians(df['yaw'].values)
    pitch_rad = np.radians(df['pitch'].values)
    
    # 1. Angular Velocity (Spherical Great-Circle)
    angular_velocity, _, _ = compute_spherical_angular_velocity(pitch_rad, yaw_rad, dt=dt)
    
    # Thesis Reference: Chapter 3, Equation (7) & Table 5 — Speed-Derived Scalar Angular Jerk
    # j_t = d(alpha_t)/dt = d^2(omega_t)/dt^2
    # Differentiates scalar angular speed omega_t twice over time. It is a speed-derived scalar
    # kinematic proxy reflecting Flash & Hogan's smoothness principle by penalizing instantaneous
    # angular acceleration changes and robotic step transitions without requiring unobservable 3D limb coordinates.
    angular_accel = np.gradient(angular_velocity, dt)
    angular_jerk = np.gradient(angular_accel, dt)
    
    # 3. Spherical Trajectory Curvature
    curvature = compute_spherical_curvature(pitch_rad, yaw_rad, dt=dt)
    
    # 4. Windowed Curvature Shannon Entropy
    curvature_entropy = calculate_windowed_entropy(curvature, window_size=32, num_bins=10)
    
    # Thesis Reference: Chapter 3, Table 5 — View Angles in Radians
    df['yaw'] = wrap_angle_rad(yaw_rad)
    df['pitch'] = np.clip(pitch_rad, -np.pi / 2.0, np.pi / 2.0)
    df['angular_velocity'] = angular_velocity
    df['angular_accel'] = angular_accel
    df['angular_jerk'] = angular_jerk
    df['trajectory_curvature'] = curvature
    df['curvature_entropy'] = curvature_entropy
    
    # 5. 8-12 Hz Physiological Hand Tremor Band Power
    if extract_tremor:
        tremor_power = compute_tremor_band_power(angular_velocity, sampling_rate=tick_rate, window_size=64)
        df['tremor_power_8_12hz'] = tremor_power
        
    return df
