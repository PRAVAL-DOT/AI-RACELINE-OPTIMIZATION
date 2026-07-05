"""
main.py
-------
Entry point for the F1 Ideal Lap Profile pipeline.
"""

import argparse
import sys
import os
import numpy as np

# Allow running from repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import telemetry as tel_mod
import ideal_profile as ip_mod
import visualization as viz_mod
import driver_analysis as da_mod
import driver_visualization as dv_mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ideal F1 lap profile from FastF1 telemetry."
    )
    parser.add_argument("--year",    type=int, default=2025)
    parser.add_argument("--gp",      type=str, default="Bahrain")
    parser.add_argument("--session", type=str, default="Q")
    parser.add_argument("--cache",   type=str, default="cache")
    parser.add_argument("--no-cache", action="store_true", help="Disable FastF1 caching")
    parser.add_argument("--driver",  type=str, default=None, help="Driver to compare against ideal (e.g. VER)")
    parser.add_argument("--output",  type=str, default=None)  # 🔥 changed
    parser.add_argument("--n-points",type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 🔥 Create dynamic output path (clean + no duplication)
    if args.output is None:
        args.output = os.path.join(
            "outputs",
            f"{args.gp}_{args.year}_ideal_profile.png"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 55)
    print(f"  F1 Ideal Lap Profile Generator")
    print(f"  {args.gp} {args.year} — {args.session}")
    print("=" * 55)

    # ── 1. Cache + session ─────────────────────────────────────
    print("\n[1/6] Loading session...")
    if not args.no_cache:
        tel_mod.enable_cache(args.cache)
    else:
        print("  (Caching disabled)")
    session = tel_mod.load_session(args.year, args.gp, args.session)

    # ── 2. Elite laps ──────────────────────────────────────────
    print("[2/6] Selecting elite laps...")
    elite_laps  = tel_mod.get_elite_laps(session)
    fastest_lap = elite_laps.iloc[0]

    print(f"  Top {len(elite_laps)} laps selected. "
          f"Fastest: {fastest_lap['Driver']} "
          f"({fastest_lap['LapTimeSeconds']:.3f}s)")

    # ── 3. Telemetry extraction ────────────────────────────────
    print("[3/6] Extracting telemetry...")
    telemetry_df = tel_mod.extract_telemetry(elite_laps)

    # ── 4. Resampling ──────────────────────────────────────────
    print("[4/6] Resampling to common distance grid...")
    fastest_tel      = fastest_lap.get_telemetry()
    reference_length = float(fastest_tel["Distance"].max())

    print(f"  Reference lap length: {reference_length:.1f}m")

    resampled = tel_mod.resample_telemetry(
        telemetry_df,
        reference_length,
        args.n_points
    )

    print(f"  Resampled {len(resampled['speed'])} laps at {args.n_points} pts")

    # ── 5. Ideal profile + corners ─────────────────────────────
    print("[5/6] Building ideal profile + detecting corners...")
    ideal_profile = ip_mod.build_ideal_profile(resampled)
    corners       = ip_mod.detect_corners(
        session,
        ideal_profile,
        resampled["distance_grid"]
    )

    print(f"  Detected {len(corners)} corners")

    s1_end_m, s2_end_m = tel_mod.get_sector_distances(fastest_lap)
    print(f"  Sector splits: S1 ends {s1_end_m:.0f}m, S2 ends {s2_end_m:.0f}m")

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

    # ── 6. Plot ────────────────────────────────────────────────
    print(f"\n[6/6] Rendering dashboard → {args.output}")

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
        save_path     = args.output,   # ✅ ONLY save path
    )

    if args.driver is not None:
        print(f"\n[7] Phase 2: Analyzing driver {args.driver}...")
        driver_laps = elite_laps[elite_laps["Driver"] == args.driver]
        if driver_laps.empty:
            print(f"  [warn] Driver {args.driver} not found in elite laps. Skipping comparison.")
        else:
            drv_lap_id = driver_laps.index[0]
            if drv_lap_id in resampled["lap_ids"]:
                drv_idx = resampled["lap_ids"].index(drv_lap_id)
                drv_speed = resampled["speed"][drv_idx]
                drv_throttle = resampled["throttle"][drv_idx]
                drv_brake = resampled["brake"][drv_idx]
                
                # Analysis
                delta_trace = da_mod.compute_time_delta(drv_speed, ideal_profile["speed"], grid)
                corner_analysis = da_mod.analyze_corners(
                    drv_speed, drv_throttle, drv_brake,
                    ideal_profile, grid, corners
                )
                
                comp_save_path = os.path.join("outputs", f"{args.gp}_{args.year}_{args.driver}_comparison.png")
                dv_mod.plot_driver_comparison(
                    grid, ideal_profile, drv_speed, delta_trace, corner_analysis,
                    corners, sector_bounds, grid[s1_end_idx], grid[s2_end_idx],
                    args.driver, comp_save_path
                )
            else:
                print(f"  [warn] Telemetry for {args.driver} was dropped during resampling.")

    print("\nDone.")


if __name__ == "__main__":
    main()