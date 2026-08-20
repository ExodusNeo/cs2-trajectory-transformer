"""Data loading and Active Tracking Window extraction package."""
from .demo_parser import CS2DemoParser
from .atw_filter import (
    calculate_relative_angle,
    find_fov_encounters,
    find_combat_event_windows,
    merge_overlapping_windows,
    extract_active_tracking_windows
)

__all__ = [
    'CS2DemoParser',
    'calculate_relative_angle',
    'find_fov_encounters',
    'find_combat_event_windows',
    'merge_overlapping_windows',
    'extract_active_tracking_windows'
]
