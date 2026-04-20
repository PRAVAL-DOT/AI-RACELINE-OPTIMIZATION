"""
telemetry.py
------------
Handles loading, extraction, and resampling of FastF1 telemetry data.
"""

import fastf1
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def enable_cache(cache_path: str = "cache") -> None:
    """Enable FastF1 disk cache to avoid re-downloading data."""
    import os
    os.makedirs(cache_path, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path)


def load_session(year: int, gp: str, session_type: str = "Q") -> fastf1.core.Session:
    """
    Load a FastF1 session.

    Parameters
    ----------
    year         : Season year, e.g. 2025
    gp           : Grand Prix name, e.g. 'Bahrain'
    session_type : 'Q' (qualifying), 'R' (race), 'FP1' etc.

    Returns
    -------
    Loaded FastF1 Session object.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def get_elite_laps(session: fastf1.core.Session, top_fraction: float = 0.10, min_laps: int = 5) -> pd.DataFrame:
    """
    Return the fastest N laps from a session.

    Parameters
    ----------
    session      : Loaded FastF1 Session
    top_fraction : Fraction of laps to keep (default = top 10%)
    min_laps     : Minimum number of laps to always keep

    Returns
    -------
    DataFrame of elite laps sorted by lap time (fastest first).
    """
    laps = session.laps.pick_quicklaps()
    laps = laps.dropna(subset=["LapTime"]).copy()
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps_sorted = laps.sort_values("LapTimeSeconds")

    top_n = max(min_laps, int(top_fraction * len(laps_sorted)))
    return laps_sorted.head(top_n)


def extract_telemetry(elite_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Pull telemetry for each lap and return a combined DataFrame.
    Uses FastF1's built-in Distance column (real metres, ~0–5400m for Bahrain).

    Parameters
    ----------
    elite_laps : DataFrame from get_elite_laps()

    Returns
    -------
    Combined telemetry DataFrame with a 'Lap' index column.
    """
    required_cols = ["Distance", "X", "Y", "Speed", "Throttle", "Brake"]
    records = []

    for idx, lap in elite_laps.iterlaps():
        try:
            tel = lap.get_telemetry().copy()
            available = [c for c in required_cols if c in tel.columns]
            tel = tel[available]

            if "Distance" not in tel.columns or len(tel) < 10:
                continue

            tel["Lap"] = idx
            records.append(tel)
        except Exception as exc:
            print(f"  [warn] Skipping lap {idx}: {exc}")

    if not records:
        raise ValueError("No usable telemetry found in the selected laps.")

    return pd.concat(records, ignore_index=True)


def get_sector_distances(fastest_lap: pd.Series) -> tuple[float, float]:
    """
    Derive exact sector split distances (metres) from the fastest lap's telemetry.
    Much more accurate than hardcoded values — works for any circuit.

    Parameters
    ----------
    fastest_lap : First row of elite_laps DataFrame

    Returns
    -------
    (sector1_end_m, sector2_end_m)
    """
    tel = fastest_lap.get_telemetry()
    s1_time = fastest_lap["Sector1Time"]
    s2_time = fastest_lap["Sector1Time"] + fastest_lap["Sector2Time"]

    s1_end = tel.loc[tel["Time"] >= s1_time, "Distance"].iloc[0]
    s2_end = tel.loc[tel["Time"] >= s2_time, "Distance"].iloc[0]
    return float(s1_end), float(s2_end)


def resample_telemetry(
    telemetry_df: pd.DataFrame,
    reference_length: float,
    n_points: int = 3000,
) -> dict:
    """
    Resample all laps onto a common distance grid for comparison.

    Parameters
    ----------
    telemetry_df     : Output of extract_telemetry()
    reference_length : Track length in metres (from fastest lap Distance max)
    n_points         : Number of equally-spaced distance points (default 3000)
                       3000 pts over ~5400m = ~1.8m resolution — fine enough
                       to resolve Bahrain's chicanes (~120m apart).

    Returns
    -------
    dict with keys:
        'distance_grid'  : np.ndarray shape (n_points,)
        'speed'          : np.ndarray shape (n_laps, n_points)
        'throttle'       : np.ndarray shape (n_laps, n_points)
        'brake'          : np.ndarray shape (n_laps, n_points)
    """
    grid = np.linspace(0, reference_length, n_points)
    speeds, throttles, brakes = [], [], []

    for lap_id in telemetry_df["Lap"].unique():
        lap_df = (
            telemetry_df[telemetry_df["Lap"] == lap_id]
            .drop_duplicates(subset="Distance")
            .sort_values("Distance")
            .copy()
        )

        if len(lap_df) < 10:
            continue

        try:
            f_spd = interp1d(lap_df["Distance"], lap_df["Speed"],
                             fill_value="extrapolate", bounds_error=False)
            f_thr = interp1d(lap_df["Distance"], lap_df["Throttle"],
                             fill_value="extrapolate", bounds_error=False)
            f_brk = interp1d(lap_df["Distance"], lap_df["Brake"],
                             fill_value="extrapolate", bounds_error=False)

            speeds.append(f_spd(grid))
            throttles.append(f_thr(grid))
            brakes.append(f_brk(grid))

        except Exception as exc:
            print(f"  [warn] Interpolation failed for lap {lap_id}: {exc}")

    if not speeds:
        raise ValueError("Resampling failed — no laps survived interpolation.")

    return {
        "distance_grid": grid,
        "speed":         np.array(speeds),
        "throttle":      np.array(throttles),
        "brake":         np.array(brakes),
    }
