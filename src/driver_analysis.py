"""
driver_analysis.py
------------------
Phase 2 features: Analyzes a specific driver's lap against the ideal profile,
computing time deltas and corner-by-corner deviations.
"""

import numpy as np
from ideal_profile import estimate_sector_time

def compute_time_delta(driver_speed: np.ndarray, ideal_speed: np.ndarray, distance_grid: np.ndarray) -> np.ndarray:
    """
    Compute cumulative time delta (loss) of the driver compared to the ideal profile.
    Positive delta means driver is slower (losing time).
    """
    drv_speed_ms = np.clip(driver_speed / 3.6, 1.0, None)
    idl_speed_ms = np.clip(ideal_speed / 3.6, 1.0, None)
    
    # Time to traverse each segment (using simple dt = ds / v approximation)
    ds = np.diff(distance_grid)
    dt_drv = ds / drv_speed_ms[:-1]
    dt_idl = ds / idl_speed_ms[:-1]
    
    # Cumulative time difference
    delta = np.cumsum(dt_drv - dt_idl)
    # Prepend 0 for the first point to match distance_grid shape
    return np.insert(delta, 0, 0.0)

def analyze_corners(
    driver_speed: np.ndarray, 
    driver_throttle: np.ndarray, 
    driver_brake: np.ndarray,
    ideal_profile: dict, 
    distance_grid: np.ndarray, 
    corners: list[dict]
) -> list[dict]:
    """
    Compute corner-by-corner deviations for the driver compared to the ideal profile.
    """
    ideal_speed = ideal_profile["speed"]
    ideal_throttle = ideal_profile["throttle"]
    ideal_brake = ideal_profile["brake"]
    
    analysis_results = []
    
    for c in corners:
        # Find grid indices for corner entry, apex, and exit
        entry_idx = int(np.argmin(np.abs(distance_grid - c["entry_dist"])))
        apex_idx  = int(np.argmin(np.abs(distance_grid - c["apex_dist"])))
        exit_idx  = int(np.argmin(np.abs(distance_grid - c["exit_dist"])))
        
        # 1. Time Loss
        drv_time = estimate_sector_time(driver_speed[entry_idx:exit_idx+1], distance_grid[entry_idx:exit_idx+1])
        idl_time = estimate_sector_time(ideal_speed[entry_idx:exit_idx+1], distance_grid[entry_idx:exit_idx+1])
        time_loss = drv_time - idl_time
        
        # 2. Braking Point Deviation
        # Look for first point where brake > 0.05 between entry and apex
        brake_window = slice(entry_idx, apex_idx)
        dist_window = distance_grid[brake_window]
        
        drv_brake_pts = np.where(driver_brake[brake_window] > 0.05)[0]
        idl_brake_pts = np.where(ideal_brake[brake_window] > 0.05)[0]
        
        drv_brake_dist = dist_window[drv_brake_pts[0]] if len(drv_brake_pts) > 0 else np.nan
        idl_brake_dist = dist_window[idl_brake_pts[0]] if len(idl_brake_pts) > 0 else np.nan
        
        brake_deviation = drv_brake_dist - idl_brake_dist if not (np.isnan(drv_brake_dist) or np.isnan(idl_brake_dist)) else np.nan
        
        # 3. Throttle Application Point
        # Look for first point where throttle > 0.90 between apex and exit
        throttle_window = slice(apex_idx, exit_idx)
        dist_window_thr = distance_grid[throttle_window]
        
        drv_thr_pts = np.where(driver_throttle[throttle_window] > 0.90)[0]
        idl_thr_pts = np.where(ideal_throttle[throttle_window] > 0.90)[0]
        
        drv_thr_dist = dist_window_thr[drv_thr_pts[0]] if len(drv_thr_pts) > 0 else np.nan
        idl_thr_dist = dist_window_thr[idl_thr_pts[0]] if len(idl_thr_pts) > 0 else np.nan
        
        throttle_deviation = drv_thr_dist - idl_thr_dist if not (np.isnan(drv_thr_dist) or np.isnan(idl_thr_dist)) else np.nan
        
        analysis_results.append({
            "num": c["num"],
            "time_loss": time_loss,
            "brake_deviation": brake_deviation,
            "throttle_deviation": throttle_deviation,
            "driver_brake_dist": drv_brake_dist,
            "ideal_brake_dist": idl_brake_dist,
            "driver_throttle_dist": drv_thr_dist,
            "ideal_throttle_dist": idl_thr_dist
        })
        
    return analysis_results
