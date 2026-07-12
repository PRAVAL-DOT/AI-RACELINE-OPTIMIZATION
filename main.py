"""
main.py
-------
Entry point for the F1 Ideal Lap Profile pipeline.
"""

import argparse
import sys
import os
import logging
import numpy as np
from scipy.interpolate import interp1d

# Allow running from repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import telemetry as tel_mod
import ideal_profile as ip_mod
import visualization as viz_mod
import feature_engineering as fe_mod

# Setup clean logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("F1Pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ideal F1 lap profile and extract engineered features."
    )
    parser.add_argument("--year",     type=int, default=2025)
    parser.add_argument("--gp",       type=str, default="Bahrain")
    parser.add_argument("--cache",    type=str, default="cache")
    parser.add_argument("--output",   type=str, default=None)
    parser.add_argument("--n-points", type=int, default=3000)
    parser.add_argument("--margin",   type=float, default=1.03)  # 3% pace threshold
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create dynamic output path base if not specified
    if args.output is None:
        args.output = os.path.join(
            "outputs",
            f"{args.gp}_{args.year}_ideal_profile.png"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 55)
    print(f"  F1 Ideal Lap Profile & Feature Pipeline")
    print(f"  Target Grand Prix: {args.gp} {args.year}")
    print("=" * 55)

    # ── 1. Cache Setup + Dynamic Session Selection ────────────────
    print("\n[1/7] Loading session with priority cascade (Q -> R -> Practice)...")
    tel_mod.enable_cache(args.cache)
    
    session = None
    chosen_session_type = None
    session_priority = ["Q", "R", "FP3", "FP2", "FP1"]

    for s_type in session_priority:
        try:
            logger.info(f"Attempting to load session type: {s_type}")
            session = tel_mod.load_session(args.year, args.gp, s_type)
            chosen_session_type = s_type
            break
        except Exception as e:
            logger.warning(f"Session {s_type} unavailable or failed to load. Trying next tier.")
            continue

    if session is None:
        raise ValueError(f"CRITICAL: Could not load any valid session for {args.year} {args.gp}.")

    print(f"  ✓ Locked into [{chosen_session_type}] session for analysis.")

    # ── 2. Structural & Pace Filtering ──────────────────────────
    print("\n[2/7] Filtering out out-laps, in-laps, and slow pace anomalies...")
    all_laps = session.laps
    
    # Macro Filter: Drop out/in laps and catastrophic outliers
    clean_laps = all_laps.pick_quicklaps().dropna(subset=["LapTime"])
    
    if len(clean_laps) == 0:
        raise ValueError(f"No clean baseline laps found in this session layout.")

    # Micro Filter: Keep majority true pace laps via statistical margin ceiling
    clean_laps = clean_laps.copy()
    clean_laps["LapTimeSeconds"] = clean_laps["LapTime"].dt.total_seconds()
    
    best_session_time = clean_laps["LapTimeSeconds"].min()
    max_pace_cutoff = best_session_time * args.margin
    
    pace_laps = clean_laps[clean_laps["LapTimeSeconds"] <= max_pace_cutoff]
    fastest_lap = pace_laps.sort_values(by="LapTimeSeconds").iloc[0]

    print(f"  ✓ Pace Filter: Retained {len(pace_laps)} / {len(clean_laps)} laps.")
    print(f"  Fastest Baseline: {fastest_lap['Driver']} ({fastest_lap['LapTimeSeconds']:.3f}s)")
    print(f"  Pace Cutoff Ceiling: {max_pace_cutoff:.3f}s (Margin: {args.margin})")

    # ── 3. Telemetry Extraction ────────────────────────────────
    print("\n[3/7] Extracting raw telemetry arrays...")
    telemetry_df = tel_mod.extract_telemetry(pace_laps)

    # ── 4. Resampling to Fixed Distance Grid ────────────────────
    print("\n[4/7] Resampling telemetry profiles to standard spatial grid...")
    fastest_tel = fastest_lap.get_telemetry()
    reference_length = float(fastest_tel["Distance"].max())

    print(f"  Reference track length: {reference_length:.1f}m")

    resampled = tel_mod.resample_telemetry(
        telemetry_df,
        reference_length,
        args.n_points
    )
    print(f"  ✓ Mapped {len(resampled['speed'])} laps across {args.n_points} unified coordinates.")

    # ── 5. Ideal Profile Generation & Corner Detection ─────────
    print("\n[5/7] Synthesizing ideal racing line profile + identifying corners...")
    ideal_profile = ip_mod.build_ideal_profile(resampled)
    corners = ip_mod.detect_corners(
        session,
        ideal_profile,
        resampled["distance_grid"]
    )
    print(f"  ✓ Mapped {len(corners)} circuit corners successfully.")

    s1_end_m, s2_end_m = tel_mod.get_sector_distances(fastest_lap)
    grid = resampled["distance_grid"]

    s1_end_idx = int(np.searchsorted(grid, s1_end_m))
    s2_end_idx = int(np.searchsorted(grid, s2_end_m))

    sector_times = ip_mod.compute_lap_times(
        ideal_profile,
        grid,
        s1_end_idx,
        s2_end_idx
    )

    print("\n" + "=" * 45)
    print("  IDEAL REFERENCE LAP TIMES")
    print("=" * 45)
    print(f"  Sector 1  : {ip_mod.fmt_time(sector_times['s1'])}")
    print(f"  Sector 2  : {ip_mod.fmt_time(sector_times['s2'])}")
    print(f"  Sector 3  : {ip_mod.fmt_time(sector_times['s3'])}")
    print(f"  Full Lap  : {ip_mod.fmt_time(sector_times['lap'])}")
    print(f"  Actual    : {ip_mod.fmt_time(fastest_lap['LapTimeSeconds'])}  ({fastest_lap['Driver']})")
    print("=" * 45)

    # ── 6. Feature Engineering & 3D Matrix Export ─────────────
    print("\n[6/7] Computing deep learning features and building 3D tensor...")
    features = fe_mod.compute_features(resampled)

    corner_ids = fe_mod.assign_corner_ids(grid, corners)
    corner_phase = fe_mod.assign_corner_phase(grid, corners)
    
    feature_tensor = fe_mod.build_feature_tensor(
        features,
        corner_ids,
        corner_phase,
    )
    print(f"  ✓ Multi-channel feature array compiled. Tensor shape: {feature_tensor.shape}")

    # Export binary array matrix to disk
    tensor_output_path = args.output.replace(".png", "_features.npy")
    np.save(tensor_output_path, feature_tensor)
    print(f"  ✓ Array binary saved to target disk location → {tensor_output_path}")

    # ── 7. Render Visualization Dashboard ─────────────────────
    print(f"\n[7/7] Plotting analytical performance dashboard → {args.output}")

    sector_bounds = [
        (0, grid[s1_end_idx]),
        (grid[s1_end_idx], grid[s2_end_idx]),
        (grid[s2_end_idx], grid[-1]),
    ]

    viz_mod.plot_dashboard(
        distance_grid = grid,
        resampled     = resampled,
        ideal_profile = ideal_profile,
        corners       = corners,
        sector_bounds = sector_bounds,
        sector_times  = sector_times,
        s1_end_dist   = grid[s1_end_idx],
        s2_end_dist   = grid[s2_end_idx],
        fastest_lap   = fastest_lap,
        year          = args.year,
        gp            = args.gp,
        save_path     = args.output,
    )

    print("\nPipeline execution complete. Ready for model ingest.")


if __name__ == "__main__":
    main()