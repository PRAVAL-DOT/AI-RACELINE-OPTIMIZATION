"""
ideal_profile.py
----------------
Builds the 'ideal driver' profile from resampled telemetry and
detects corners using official FastF1 circuit data.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def build_ideal_profile(resampled: dict, smooth_sigma: float = 2.0) -> dict:
    """
    Construct the ideal speed/throttle/brake profile.

    Strategy:
    - Speed    : 90th percentile across laps (fastest through each point)
    - Throttle : 90th percentile (most aggressive application)
    - Brake    : mean (braking is more symmetric across elite laps)
    - All channels smoothed with a Gaussian filter.

    Parameters
    ----------
    resampled    : Output dict from telemetry.resample_telemetry()
    smooth_sigma : Gaussian smoothing sigma in grid-point units.
                   sigma=2 at 3000pts/5400m ≈ 3.6m half-width — tight enough
                   to preserve chicane dips, loose enough to remove sensor noise.

    Returns
    -------
    dict with keys: 'speed', 'throttle', 'brake'  (all np.ndarray, 1-D)
    """
    ideal_speed    = np.percentile(resampled["speed"],    90, axis=0)
    ideal_throttle = np.percentile(resampled["throttle"], 90, axis=0)
    ideal_brake    = np.mean(resampled["brake"],              axis=0)

    return {
        "speed":    gaussian_filter1d(ideal_speed,    sigma=smooth_sigma),
        "throttle": gaussian_filter1d(ideal_throttle, sigma=smooth_sigma),
        "brake":    gaussian_filter1d(ideal_brake,    sigma=smooth_sigma),
    }


def detect_corners(
    session,
    ideal_profile: dict,
    distance_grid: np.ndarray,
    entry_exit_threshold_kmh: float = 15.0,
    min_corner_width_m: float = 30.0,
    max_expand_m: float = 150.0,
) -> list[dict]:
    """
    Detect corners using FastF1's official circuit corner data.

    This is far more robust than peak-finding on the speed trace because:
    - It uses the circuit's known corner distances, not heuristics.
    - Works for flat-out corners that produce no visible speed dip.
    - Corner numbers match the official F1 corner numbering.

    Each corner entry/exit is found by walking outward from the official
    apex distance until speed rises by `entry_exit_threshold_kmh` km/h.

    Parameters
    ----------
    session                   : Loaded FastF1 Session (for circuit_info)
    ideal_profile             : Output of build_ideal_profile()
    distance_grid             : np.ndarray of distance points (metres)
    entry_exit_threshold_kmh  : How far above apex speed to mark entry/exit
    min_corner_width_m        : Minimum visual width for flat-out corners
    max_expand_m              : Max distance to walk left/right from apex

    Returns
    -------
    List of dicts, one per corner:
        num, entry_dist, apex_dist, exit_dist, apex_speed (all in metres / km/h)
    """
    circuit_info  = session.get_circuit_info()
    official_corners = circuit_info.corners
    speed = ideal_profile["speed"]
    corners = []

    for _, row in official_corners.iterrows():
        c_dist = row["Distance"]
        c_num  = row["Number"]

        # Find closest grid index to the official corner distance
        apex_idx   = int(np.argmin(np.abs(distance_grid - c_dist)))
        apex_speed = speed[apex_idx]
        cutoff     = apex_speed + entry_exit_threshold_kmh

        # Expand left → entry (braking zone start)
        entry = apex_idx
        while (
            entry > 0
            and speed[entry] < cutoff
            and (distance_grid[apex_idx] - distance_grid[entry]) < max_expand_m
        ):
            entry -= 1

        # Expand right → exit (throttle application point)
        exit_ = apex_idx
        while (
            exit_ < len(speed) - 1
            and speed[exit_] < cutoff
            and (distance_grid[exit_] - distance_grid[apex_idx]) < max_expand_m
        ):
            exit_ += 1

        # Flat-out corners: enforce minimum visual width so they show on plot
        if distance_grid[exit_] - distance_grid[entry] < min_corner_width_m:
            pts = int(min_corner_width_m / (distance_grid[1] - distance_grid[0]) / 2)
            entry = max(0, apex_idx - pts)
            exit_ = min(len(distance_grid) - 1, apex_idx + pts)

        corners.append({
            "num":        int(c_num),
            "entry_dist": float(distance_grid[entry]),
            "apex_dist":  float(distance_grid[apex_idx]),
            "exit_dist":  float(distance_grid[exit_]),
            "apex_speed": float(apex_speed),
        })

    # Sort corners by distance and prevent overlaps
    corners = sorted(corners, key=lambda x: x["apex_dist"])
    for i in range(len(corners) - 1):
        if corners[i]["exit_dist"] > corners[i+1]["entry_dist"]:
            midpoint = (corners[i]["exit_dist"] + corners[i+1]["entry_dist"]) / 2.0
            corners[i]["exit_dist"] = midpoint
            corners[i+1]["entry_dist"] = midpoint

    return corners


def estimate_sector_time(speed_kmh: np.ndarray, dist_m: np.ndarray) -> float:
    """
    Estimate sector lap time by integrating dt = ds / v.

    This is a kinematic approximation — it ignores instantaneous acceleration
    forces but is accurate to ~0.3–0.8s for reference purposes.

    Parameters
    ----------
    speed_kmh : Speed array in km/h
    dist_m    : Corresponding distance array in metres

    Returns
    -------
    Estimated sector time in seconds.
    """
    speed_ms = np.clip(speed_kmh / 3.6, 1.0, None)  # avoid div/0
    # Use Trapezoidal rule for more accurate integration: dt = ds / v
    t = np.trapezoid(1.0 / speed_ms, dist_m)
    return float(t)


def compute_lap_times(
    ideal_profile: dict,
    distance_grid: np.ndarray,
    s1_end_idx: int,
    s2_end_idx: int,
) -> dict:
    """
    Compute ideal sector and lap times.

    Parameters
    ----------
    ideal_profile : Output of build_ideal_profile()
    distance_grid : 1-D np.ndarray
    s1_end_idx    : Grid index where Sector 1 ends
    s2_end_idx    : Grid index where Sector 2 ends

    Returns
    -------
    dict with keys: 's1', 's2', 's3', 'lap'  (all floats, seconds)
    """
    spd = ideal_profile["speed"]
    t_s1 = estimate_sector_time(spd[:s1_end_idx],   distance_grid[:s1_end_idx])
    t_s2 = estimate_sector_time(spd[s1_end_idx:s2_end_idx], distance_grid[s1_end_idx:s2_end_idx])
    t_s3 = estimate_sector_time(spd[s2_end_idx:],   distance_grid[s2_end_idx:])
    return {"s1": t_s1, "s2": t_s2, "s3": t_s3, "lap": t_s1 + t_s2 + t_s3}


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS.mmm string."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"
