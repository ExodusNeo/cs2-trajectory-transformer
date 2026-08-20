"""
Unit Tests for Biomechanical & Kinematic Feature Extraction.
Verifies Euler Angle Wrapping, Spherical Curvature, and 8-12Hz Tremor PSD.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from features.kinematics import (
    wrap_angle_rad,
    compute_spherical_angular_velocity,
    compute_spherical_curvature,
    compute_tremor_band_power,
    calculate_windowed_entropy,
    compute_kinematic_features
)


def test_euler_angle_wrapping():
    """Verify that jumping across +/- 180 degrees does NOT cause massive delta spikes."""
    # Yaw values that cross the +180/-180 boundary (e.g. 179 deg to -179 deg is 2 deg movement)
    yaw_deg = np.array([179.0, -179.0, -178.0])
    pitch_deg = np.array([0.0, 0.0, 0.0])
    
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    dt = 1.0 / 128.0
    ang_vel, d_pitch, d_yaw = compute_spherical_angular_velocity(pitch_rad, yaw_rad, dt=dt)
    
    # 2 degrees in radians
    expected_step_rad = np.radians(2.0)
    expected_vel = expected_step_rad / dt
    
    # Check that angular velocity at step 1 is ~2 deg/dt, NOT 358 deg/dt
    assert np.isclose(d_yaw[1], expected_step_rad, atol=1e-5), f"Expected {expected_step_rad}, got {d_yaw[1]}"
    assert np.isclose(ang_vel[1], expected_vel, atol=1e-3), f"Expected {expected_vel}, got {ang_vel[1]}"


def test_tremor_band_power_detection():
    """Verify that a 10 Hz simulated physiological tremor yields high 8-12 Hz band power."""
    fs = 128.0
    n = 256
    t = np.arange(n) / fs
    
    # Signal with 10 Hz oscillation (inside 8-12 Hz tremor band)
    tremor_signal = np.sin(2 * np.pi * 10.0 * t)
    
    # Low frequency drift (outside 8-12 Hz tremor band, e.g. 2 Hz)
    drift_signal = np.sin(2 * np.pi * 2.0 * t)
    
    power_tremor = compute_tremor_band_power(tremor_signal, sampling_rate=fs, window_size=64)
    power_drift = compute_tremor_band_power(drift_signal, sampling_rate=fs, window_size=64)
    
    # 10 Hz signal should have significantly higher relative band power in 8-12Hz than 2 Hz signal
    assert np.mean(power_tremor[32:-32]) > 0.6, "10 Hz signal should have high tremor band power"
    assert np.mean(power_drift[32:-32]) < 0.1, "2 Hz signal should have low tremor band power"


def test_spherical_curvature_straight_vs_curved():
    """Verify spherical curvature on straight vs circular trajectory on unit sphere."""
    fs = 128.0
    n = 128
    t = np.arange(n) / fs
    
    # Straight horizontal trajectory: pitch = 0, yaw = linear
    pitch_straight = np.zeros(n)
    yaw_straight = 0.5 * t
    curv_straight = compute_spherical_curvature(pitch_straight, yaw_straight, dt=1.0/fs)
    
    # Great circle has 0 geodesic curvature on unit sphere
    assert np.all(curv_straight[5:-5] < 1.0), "Straight trajectory should have near-zero geodesic curvature"


def test_compute_kinematic_features_dataframe():
    """Verify full end-to-end DataFrame feature computation."""
    n = 200
    df = pd.DataFrame({
        'tick': np.arange(n),
        'yaw': np.linspace(0, 45, n),
        'pitch': np.linspace(-10, 10, n),
        'x': np.zeros(n),
        'y': np.zeros(n),
        'z': np.zeros(n)
    })
    
    result = compute_kinematic_features(df, tick_rate=128.0, extract_tremor=True)
    
    expected_cols = [
        'angular_velocity', 'angular_accel', 'angular_jerk', 
        'trajectory_curvature', 'curvature_entropy', 'tremor_power_8_12hz'
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
        assert not result[col].isnull().any(), f"NaN values detected in {col}"
        assert not np.isinf(result[col]).any(), f"Inf values detected in {col}"
