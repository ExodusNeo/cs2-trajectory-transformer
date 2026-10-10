"""
Biomechanical & Kinematic Feature Extraction Module for CS2 / FPS Replay Telemetry.

Extracts candidate motor-control features from 64 Hz tick-sampled view angles:
1. Shortest-path Euler Angle Differences (Wrapped in [-pi, pi])
2. Great-Circle Spherical Angular Velocity (rad/s)
3. Speed-derived Angular Acceleration (rad/s^2) and Jerk (rad/s^3)
4. Intrinsic Geodesic Curvature of the sight vector on S^2
5. 8-12 Hz Relative Band Power of the signed angular-rate components
6. Sliding-window Curvature Shannon Entropy
7. Target-relative aim error to the nearest enemy head and its rate

All channels are *candidate* features whose discriminative value is tested empirically
(ablation study). None of them is assumed to be an invariant that aim assistance cannot fake.

Known measurement limit: small CS2 replay view-angle steps cluster at one mouse count
(0.022 deg x sensitivity), and more than half of live ticks show zero angular change.
Physiological tremor is usually smaller than one count, so the tremor channel may carry
little signal on real data (see probe_replay_signal.py).
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ==============================================================================
# Canonical model input contract (single source of truth for every script).
# Absolute yaw is excluded on purpose: it encodes map orientation, which lets a model
# learn map position instead of aiming behaviour. It is still stored in the Parquet
# output (wrapped radians) for the raw-angle baseline in ablation.py.
# ==============================================================================
MODEL_FEATURE_COLUMNS: List[str] = [
    'pitch',
    'angular_velocity',
    'angular_accel',
    'angular_jerk',
    'trajectory_curvature',
    'curvature_entropy',
    'tremor_power_8_12hz',
    'aim_error',
    'aim_error_rate',
]

# Approximate standing eye height above the player origin in Hammer units.
EYE_HEIGHT_UNITS: float = 64.0


def wrap_angle_rad(angles: np.ndarray) -> np.ndarray:
    """
    Wraps angular differences to the interval [-pi, pi].

    Thesis Reference: Chapter 3, Equation (4) — Shortest-Path Euler Angle Wrapping.
    Eliminates artificial 358-degree jumps across the +/-180 degree yaw boundary.
    """
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def compute_spherical_angular_velocity(
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
    dt: float = 1.0 / 64.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes great-circle angular speed of the view direction on the unit sphere.

    Thesis Reference: Chapter 3, Equation (5)
    omega_t = (1 / dt) * sqrt(wrap(dp)^2 + (cos(p) * wrap(dy))^2)

    Parameters
    ----------
    pitch_rad, yaw_rad : np.ndarray
        View angles in radians, shape (N,).
    dt : float
        Tick interval in seconds (1/64 s).

    Returns
    -------
    (angular_velocity [rad/s], d_pitch [rad], d_yaw [rad]), each shape (N,).
    """
    d_yaw = wrap_angle_rad(np.diff(yaw_rad, prepend=yaw_rad[0]))
    d_pitch = wrap_angle_rad(np.diff(pitch_rad, prepend=pitch_rad[0]))
    arc_diff = np.sqrt(d_pitch ** 2 + (np.cos(pitch_rad) * d_yaw) ** 2)
    return arc_diff / dt, d_pitch, d_yaw


def compute_signed_angular_rates(
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
    dt: float = 1.0 / 64.0
) -> np.ndarray:
    """
    Signed tangent-plane angular-rate components of the view direction.

    Unlike the speed magnitude omega_t, these components keep their sign, so an
    oscillation at f Hz stays at f Hz (rectifying it with |.| would move its power to 2f).

    Returns
    -------
    np.ndarray, shape (N, 2): [cos(pitch) * d_yaw/dt, d_pitch/dt] in rad/s.
    """
    _, d_pitch, d_yaw = compute_spherical_angular_velocity(pitch_rad, yaw_rad, dt=dt)
    return np.stack([np.cos(pitch_rad) * d_yaw / dt, d_pitch / dt], axis=-1)


def compute_tremor_band_power(
    signal: np.ndarray,
    sampling_rate: float = 64.0,
    window_size: int = 64,
    tremor_low: float = 8.0,
    tremor_high: float = 12.0,
    static_power_floor: float = 1e-4
) -> np.ndarray:
    """
    Sliding-window relative band power in the 8-12 Hz physiological tremor band.

    Thesis Reference: Chapter 3, Equation (11) — Tremor Band Power (TBP)
    TBP = (sum_{f=8}^{12} |X(f)|^2) / (sum_{f=1}^{30} |X(f)|^2 + eps)

    Each tick t uses the Hann-windowed window [max(0, t - W/2), +W), edge-padded at the
    end of the sequence. For multi-channel input the power spectra are summed over channels.

    Parameters
    ----------
    signal : np.ndarray
        Shape (N,) or (N, C). Use signed rates (compute_signed_angular_rates), not |omega|.
    sampling_rate : float
        Hz (64.0 for CS2 replays).
    window_size : int
        FFT length in ticks (64 ticks = 1.0 s, 1 Hz bins).
    static_power_floor : float
        Windows whose 1-30 Hz power is below this are reported as 0.0 (static view).

    Returns
    -------
    np.ndarray, shape (N,), ratio in [0.0, 1.0].
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    # Right edge-pad so every tick owns a full-length window (matches the reference loop).
    padded = np.pad(x, ((0, window_size), (0, 0)), mode='edge')
    starts = np.maximum(0, np.arange(n) - window_size // 2)
    windows = sliding_window_view(padded, window_size, axis=0)[starts]  # (N, C, W)

    windows = windows - windows.mean(axis=-1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(window_size), axis=-1)) ** 2
    spectrum = spectrum.sum(axis=1)  # sum over channels -> (N, F)

    freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)
    tremor_mask = (freqs >= tremor_low) & (freqs <= tremor_high)
    total_mask = (freqs >= 1.0) & (freqs <= 30.0)

    total_p = spectrum[:, total_mask].sum(axis=1)
    tremor_p = spectrum[:, tremor_mask].sum(axis=1)
    tbp = np.where(total_p < static_power_floor, 0.0, tremor_p / (total_p + 1e-9))
    return tbp.astype(np.float32)


def compute_spherical_curvature(
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
    dt: float = 1.0 / 64.0,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Intrinsic geodesic curvature kappa_g of the unit sight vector v(t) on S^2.

    Thesis Reference: Chapter 3, Equation (9)
        kappa_g(t) = |v . (v' x v'')| / (||v'||^3 + eps)

    Geometry note: great circles have kappa_g = 0. A purely horizontal mouse swipe changes
    only yaw and traces a circle of latitude, whose kappa_g = |tan(pitch)|, so ordinary
    mouse motion is not geodesic away from the horizon. Interpret this channel empirically,
    not as "humans follow geodesics".

    Returns
    -------
    np.ndarray, shape (N,), dimensionless, clipped to [0, 50].
    """
    v = np.stack([
        np.cos(pitch_rad) * np.cos(yaw_rad),
        np.cos(pitch_rad) * np.sin(yaw_rad),
        np.sin(pitch_rad),
    ], axis=-1)
    v_prime = np.gradient(v, dt, axis=0)
    v_double_prime = np.gradient(v_prime, dt, axis=0)

    speed = np.linalg.norm(v_prime, axis=-1)
    scalar_triple = np.abs(np.sum(v * np.cross(v_prime, v_double_prime), axis=-1))

    moving = speed >= 1e-3
    curvature = np.zeros_like(speed)
    curvature[moving] = scalar_triple[moving] / (speed[moving] ** 3 + eps)
    return np.clip(curvature, 0.0, 50.0)


def calculate_windowed_entropy(series: np.ndarray, window_size: int = 32, num_bins: int = 10) -> np.ndarray:
    """
    Sliding-window Shannon entropy (bits) of a trajectory metric.

    Thesis Reference: Chapter 3, Equation (10) — Curvature Shannon Entropy.
    Returns np.ndarray, shape (N,), in [0, log2(num_bins)].
    """
    entropy_list = np.zeros(len(series), dtype=np.float32)
    half_w = window_size // 2
    n = len(series)

    for i in range(n):
        sub_window = series[max(0, i - half_w):min(n, i + half_w)]
        hist, _ = np.histogram(sub_window, bins=num_bins, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            prob = hist / np.sum(hist)
            entropy_list[i] = -np.sum(prob * np.log2(prob))

    return entropy_list


def view_target_angles_rad(
    eye_pos: np.ndarray,
    pitch_deg: np.ndarray,
    yaw_deg: np.ndarray,
    target_pos: np.ndarray
) -> np.ndarray:
    """
    Vectorized angle between the view direction and the line of sight to a target.

    Thesis Reference: Chapter 3, Equation (3) — FOV deviation angle.
    CS2 convention: yaw 0 = +X, 90 = +Y; positive pitch looks DOWN (+Z is up).

    Parameters
    ----------
    eye_pos, target_pos : np.ndarray, shape (N, 3), Hammer units.
    pitch_deg, yaw_deg : np.ndarray, shape (N,), degrees.

    Returns
    -------
    np.ndarray, shape (N,), radians in [0, pi].
    """
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    look = np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)], axis=-1)
    rel = target_pos - eye_pos
    # atan2(|a x b|, a . b) stays accurate near 0 rad, where arccos(dot) loses ~1e-6 rad of
    # precision. Small aim errors are exactly the regime that matters here.
    cross_norm = np.linalg.norm(np.cross(look, rel), axis=-1)
    dot = np.sum(look * rel, axis=-1)
    return np.arctan2(cross_norm, dot)


def compute_aim_error(
    player_df: pd.DataFrame,
    enemy_df: Optional[pd.DataFrame],
    max_distance: float = 3500.0,
    eye_height: float = EYE_HEIGHT_UNITS
) -> np.ndarray:
    """
    Target-relative aim error: angle from the crosshair to the nearest enemy head per tick.

    This operationalizes the Fitts' Law target-acquisition view of aiming. Aim assistance
    acts on the crosshair-target relationship, which self-kinematics alone cannot see.
    Limitations: no map occlusion (no BSP geometry in demos) and a fixed standing eye height
    (crouching enemies are about 18 units lower).

    Parameters
    ----------
    player_df : DataFrame with 'tick', 'X', 'Y', 'Z', 'pitch', 'yaw' (degrees).
    enemy_df : DataFrame with 'tick', 'X', 'Y', 'Z' (and optional 'is_alive') for all enemies.

    Returns
    -------
    np.ndarray, shape (len(player_df),), radians in [0, pi]; pi where no enemy is in range.
    """
    n = len(player_df)
    if enemy_df is None or enemy_df.empty or n == 0:
        return np.full(n, np.pi, dtype=np.float64)

    enemies = enemy_df
    if 'is_alive' in enemies.columns:
        enemies = enemies[enemies['is_alive'].astype(bool)]
    merged = pd.merge(
        player_df[['tick', 'X', 'Y', 'Z', 'pitch', 'yaw']],
        enemies[['tick', 'X', 'Y', 'Z']],
        on='tick',
        suffixes=('_p', '_e'),
    )
    if merged.empty:
        return np.full(n, np.pi, dtype=np.float64)

    eye = merged[['X_p', 'Y_p', 'Z_p']].to_numpy(dtype=np.float64) + [0.0, 0.0, eye_height]
    head = merged[['X_e', 'Y_e', 'Z_e']].to_numpy(dtype=np.float64) + [0.0, 0.0, eye_height]
    dist = np.linalg.norm(head - eye, axis=-1)
    # dist > 1 unit excludes degenerate zero-length sight lines (a player paired with themself).
    merged['aim_error'] = np.where(
        (dist > 1.0) & (dist <= max_distance),
        view_target_angles_rad(eye, merged['pitch'].to_numpy(), merged['yaw'].to_numpy(), head),
        np.pi,
    )
    per_tick = merged.groupby('tick')['aim_error'].min()
    return per_tick.reindex(player_df['tick'].to_numpy()).fillna(np.pi).to_numpy(dtype=np.float64)


def compute_kinematic_features(
    df: pd.DataFrame,
    tick_rate: float = 64.0,
    extract_tremor: bool = True,
    aim_error: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Transforms raw pitch/yaw view angles into the candidate motor-control channels.

    Parameters
    ----------
    df : pd.DataFrame
        Contains 'yaw' and 'pitch' in degrees (one ATW, contiguous ticks).
    tick_rate : float
        Sampling frequency in Hz (64.0).
    extract_tremor : bool
        Whether to compute 'tremor_power_8_12hz'.
    aim_error : np.ndarray, optional
        Per-tick target aim error in radians (from compute_aim_error). If omitted, an
        existing 'aim_error' column is used; otherwise it is filled with pi (no target info).

    Returns
    -------
    The same DataFrame with yaw/pitch converted to radians and these columns added:
    angular_velocity (rad/s), angular_accel (rad/s^2), angular_jerk (rad/s^3),
    trajectory_curvature, curvature_entropy (bits), tremor_power_8_12hz (ratio),
    aim_error (rad), aim_error_rate (rad/s).
    """
    dt = 1.0 / tick_rate
    yaw_rad = np.radians(df['yaw'].to_numpy(dtype=np.float64))
    pitch_rad = np.radians(df['pitch'].to_numpy(dtype=np.float64))

    angular_velocity, _, _ = compute_spherical_angular_velocity(pitch_rad, yaw_rad, dt=dt)

    # Thesis Reference: Chapter 3, Equations (6)-(7) — Speed-Derived Scalar Angular Jerk
    # j_t = d^2(omega_t)/dt^2. A scalar smoothness proxy, not 3D limb jerk. On 64 Hz angles whose
    # small steps are single mouse counts, small corrections are dominated by quantization noise.
    angular_accel = np.gradient(angular_velocity, dt)
    angular_jerk = np.gradient(angular_accel, dt)

    curvature = compute_spherical_curvature(pitch_rad, yaw_rad, dt=dt)
    curvature_entropy = calculate_windowed_entropy(curvature, window_size=32, num_bins=10)

    if aim_error is None:
        aim_error = df['aim_error'].to_numpy(dtype=np.float64) if 'aim_error' in df.columns else np.full(len(df), np.pi)
    aim_error = np.asarray(aim_error, dtype=np.float64)
    aim_error_rate = np.gradient(aim_error, dt) if len(aim_error) > 1 else np.zeros_like(aim_error)
    # Ticks next to "no target" (pi) carry an artificial jump, not motion.
    no_target = aim_error >= np.pi - 1e-9
    near_no_target = no_target.copy()
    near_no_target[1:] |= no_target[:-1]
    near_no_target[:-1] |= no_target[1:]
    aim_error_rate[near_no_target] = 0.0

    # Thesis Reference: Chapter 3, Table 5 — View Angles in Radians
    df['yaw'] = wrap_angle_rad(yaw_rad)
    df['pitch'] = np.clip(pitch_rad, -np.pi / 2.0, np.pi / 2.0)
    df['angular_velocity'] = angular_velocity
    df['angular_accel'] = angular_accel
    df['angular_jerk'] = angular_jerk
    df['trajectory_curvature'] = curvature
    df['curvature_entropy'] = curvature_entropy
    df['aim_error'] = aim_error
    df['aim_error_rate'] = aim_error_rate

    if extract_tremor:
        rates = compute_signed_angular_rates(pitch_rad, yaw_rad, dt=dt)
        df['tremor_power_8_12hz'] = compute_tremor_band_power(rates, sampling_rate=tick_rate, window_size=64)

    return df
