"""
visualization.py
----------------
All plotting logic for the ideal driving profile dashboard.
Kept separate from analysis logic so charts can be regenerated
without re-running the data pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ideal_profile import fmt_time


# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG       = "#0f0f0f"
GRID_COL      = "#1e1e1e"
TEXT_COL      = "#cccccc"
SPEED_COL     = "#00c8ff"
THROTTLE_COL  = "#44ff44"
BRAKE_COL     = "#ff4444"
CORNER_COL    = "#ff6600"
APEX_COL      = "#ff3333"
APEX_LBL_COL  = "#ff9966"
SECTOR_COLS   = ["#1a2a1a", "#1a1a2a", "#2a1a1a"]   # green / blue / red tint


def _style_ax(ax: plt.Axes) -> None:
    """Apply consistent dark-theme styling to an axes object."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.yaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)


def _add_sector_shading(ax: plt.Axes, sector_bounds: list[tuple]) -> None:
    for i, (s_start, s_end) in enumerate(sector_bounds):
        ax.axvspan(s_start, s_end, alpha=0.10, color=SECTOR_COLS[i], zorder=1)


def _add_sector_dividers(ax: plt.Axes, s1_dist: float, s2_dist: float) -> None:
    for d in [s1_dist, s2_dist]:
        ax.axvline(d, color="#555555", linewidth=1, linestyle="--")


def _add_corner_shading(ax: plt.Axes, corners: list[dict]) -> None:
    for c in corners:
        ax.axvspan(c["entry_dist"], c["exit_dist"],
                   alpha=0.15, color=CORNER_COL, zorder=2)


def plot_dashboard(
    distance_grid:    np.ndarray,
    resampled:        dict,
    ideal_profile:    dict,
    corners:          list[dict],
    sector_bounds:    list[tuple],
    sector_times:     dict,
    s1_end_dist:      float,
    s2_end_dist:      float,
    fastest_lap:      pd.Series,
    year:             int,
    gp:               str,
    save_path:        str = "outputs/bahrain_ideal_profile.png",
) -> plt.Figure:
    """
    Render the full 4-panel dashboard:
      Panel 0 — Speed trace + corner markers + sector time labels
      Panel 1 — Throttle trace
      Panel 2 — Brake trace
      Panel 3 — Reference lap time comparison table

    Parameters
    ----------
    distance_grid  : 1-D np.ndarray of distance values (metres)
    resampled      : dict from telemetry.resample_telemetry()
    ideal_profile  : dict from ideal_profile.build_ideal_profile()
    corners        : list of corner dicts from ideal_profile.detect_corners()
    sector_bounds  : list of (start_m, end_m) for S1, S2, S3
    sector_times   : dict {'s1', 's2', 's3', 'lap'} from compute_lap_times()
    s1_end_dist    : Distance in metres where Sector 1 ends
    s2_end_dist    : Distance in metres where Sector 2 ends
    fastest_lap    : First row of elite_laps DataFrame
    year, gp       : Used in the title string
    save_path      : Output file path

    Returns
    -------
    matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[3, 1.5, 1.5, 1.2],
        hspace=0.08,
    )

    ax_speed    = fig.add_subplot(gs[0])
    ax_throttle = fig.add_subplot(gs[1], sharex=ax_speed)
    ax_brake    = fig.add_subplot(gs[2], sharex=ax_speed)
    ax_info     = fig.add_subplot(gs[3])

    for ax in [ax_speed, ax_throttle, ax_brake]:
        _style_ax(ax)

    # ── Speed ─────────────────────────────────────────────────────────────────
    for lap in resampled["speed"]:
        ax_speed.plot(distance_grid, lap, color="#888888", alpha=0.15, linewidth=0.7)

    ax_speed.plot(distance_grid, ideal_profile["speed"],
                  color=SPEED_COL, linewidth=2.2, label="Ideal speed profile", zorder=5)

    _add_sector_shading(ax_speed, sector_bounds)
    _add_sector_dividers(ax_speed, s1_end_dist, s2_end_dist)
    _add_corner_shading(ax_speed, corners)

    # Apex dots + turn labels
    for c in corners:
        ax_speed.scatter(c["apex_dist"], c["apex_speed"],
                         color=APEX_COL, s=45, zorder=6)
        ax_speed.annotate(
            str(c["num"]),
            (c["apex_dist"], c["apex_speed"]),
            textcoords="offset points", xytext=(0, -13),
            fontsize=7, ha="center", color=APEX_LBL_COL, fontweight="bold",
        )

    # Sector time boxes
    for i, (s_start, s_end) in enumerate(sector_bounds):
        mid = (s_start + s_end) / 2
        key = ["s1", "s2", "s3"][i]
        ax_speed.text(
            mid, 310, f"S{i+1}: {fmt_time(sector_times[key])}",
            ha="center", va="top", fontsize=9,
            color="#ffffff", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=SECTOR_COLS[i],
                      edgecolor="#444444", alpha=0.9),
        )

    ax_speed.set_ylabel("Speed (km/h)", color=TEXT_COL, fontsize=10)
    ax_speed.legend(loc="upper right", fontsize=9,
                    facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT_COL)
    ax_speed.set_title(
        f"Ideal Driving Profile — {gp} {year} Q  |  "
        f"Ideal Lap: {fmt_time(sector_times['lap'])}  |  "
        f"Corners: {len(corners)}",
        color=TEXT_COL, fontsize=12, pad=10,
    )
    plt.setp(ax_speed.get_xticklabels(), visible=False)

    # ── Throttle ──────────────────────────────────────────────────────────────
    for lap in resampled["throttle"]:
        ax_throttle.plot(distance_grid, lap, color="#336633", alpha=0.15, linewidth=0.7)
    ax_throttle.plot(distance_grid, ideal_profile["throttle"],
                     color=THROTTLE_COL, linewidth=2, label="Ideal throttle", zorder=5)
    _add_sector_shading(ax_throttle, sector_bounds)
    _add_sector_dividers(ax_throttle, s1_end_dist, s2_end_dist)
    _add_corner_shading(ax_throttle, corners)
    ax_throttle.set_ylabel("Throttle (%)", color=TEXT_COL, fontsize=10)
    ax_throttle.legend(loc="upper right", fontsize=9,
                       facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT_COL)
    plt.setp(ax_throttle.get_xticklabels(), visible=False)

    # ── Brake ─────────────────────────────────────────────────────────────────
    for lap in resampled["brake"]:
        ax_brake.plot(distance_grid, lap, color="#662222", alpha=0.15, linewidth=0.7)
    ax_brake.plot(distance_grid, ideal_profile["brake"],
                  color=BRAKE_COL, linewidth=2, label="Mean brake", zorder=5)
    _add_sector_shading(ax_brake, sector_bounds)
    _add_sector_dividers(ax_brake, s1_end_dist, s2_end_dist)
    _add_corner_shading(ax_brake, corners)
    ax_brake.set_ylabel("Brake", color=TEXT_COL, fontsize=10)
    ax_brake.set_xlabel("Distance (m)", color=TEXT_COL, fontsize=10)
    ax_brake.legend(loc="upper right", fontsize=9,
                    facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT_COL)

    # ── Info table ────────────────────────────────────────────────────────────
    ax_info.set_facecolor("#0a0a0a")
    ax_info.axis("off")

    def _safe_sec(col):
        val = fastest_lap.get(col)
        return fmt_time(val.total_seconds()) if pd.notna(val) else "N/A"

    table_rows = [
        ["",                              "Sector 1", "Sector 2", "Sector 3", "Full Lap"],
        ["Ideal reference",               fmt_time(sector_times["s1"]),
                                          fmt_time(sector_times["s2"]),
                                          fmt_time(sector_times["s3"]),
                                          fmt_time(sector_times["lap"])],
        [f"Fastest ({fastest_lap['Driver']})",
                                          _safe_sec("Sector1Time"),
                                          _safe_sec("Sector2Time"),
                                          _safe_sec("Sector3Time"),
                                          fmt_time(fastest_lap["LapTimeSeconds"])],
    ]

    col_x    = [0.01, 0.22, 0.42, 0.62, 0.82]
    row_y    = [0.78, 0.45, 0.10]
    hdr_bg   = "#222233"
    row_bgs  = ["#1a1a2a", "#0f1a0f", "#1a0f0f"]

    for ri, row in enumerate(table_rows):
        bg = hdr_bg if ri == 0 else row_bgs[ri]
        ax_info.axhspan(row_y[ri] - 0.18, row_y[ri] + 0.18,
                        xmin=0, xmax=1, color=bg, alpha=0.8)
        for ci, cell in enumerate(row):
            bold  = (ri == 0 or ci == 0)
            color = "#aaaaff" if ri == 0 else ("#88ff88" if ri == 1 else "#ff8888")
            if ci == 0:
                color = TEXT_COL
            ax_info.text(
                col_x[ci] + 0.01, row_y[ri], str(cell),
                transform=ax_info.transAxes,
                fontsize=9, color=color,
                fontweight="bold" if bold else "normal",
                va="center", ha="left", fontfamily="monospace",
            )

    ax_info.text(0.5, 0.97, "Reference Lap Times",
                 transform=ax_info.transAxes,
                 fontsize=10, color="#aaaaaa",
                 ha="center", va="top", fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved → {save_path}")
    return fig
