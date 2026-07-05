"""
driver_visualization.py
-----------------------
Visualization logic for Phase 2: Driver vs Ideal comparison.
Plots delta traces, corner-by-corner breakdown, and deviations.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from visualization import (
    _style_ax, 
    _add_sector_shading, 
    _add_sector_dividers, 
    _add_corner_shading,
    DARK_BG, GRID_COL, TEXT_COL, SPEED_COL, CORNER_COL, APEX_COL, APEX_LBL_COL
)

def plot_driver_comparison(
    distance_grid: np.ndarray,
    ideal_profile: dict,
    driver_speed: np.ndarray,
    delta_trace: np.ndarray,
    corner_analysis: list[dict],
    corners: list[dict],
    sector_bounds: list[tuple],
    s1_end_dist: float,
    s2_end_dist: float,
    driver_name: str,
    save_path: str
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3, 1.5, 2],
        hspace=0.15,
    )

    ax_speed = fig.add_subplot(gs[0])
    ax_delta = fig.add_subplot(gs[1], sharex=ax_speed)
    ax_table = fig.add_subplot(gs[2])

    for ax in [ax_speed, ax_delta]:
        _style_ax(ax)

    # ── 1. Speed Trace ────────────────────────────────────────────────────────
    ax_speed.plot(distance_grid, ideal_profile["speed"], color=SPEED_COL, linewidth=2, label="Ideal Speed", alpha=0.8)
    ax_speed.plot(distance_grid, driver_speed, color="#ffdd00", linewidth=1.5, label=f"{driver_name} Speed", alpha=0.9)
    
    _add_sector_shading(ax_speed, sector_bounds)
    _add_sector_dividers(ax_speed, s1_end_dist, s2_end_dist)
    _add_corner_shading(ax_speed, corners)
    
    # Apex dots + turn labels
    for c in corners:
        ax_speed.scatter(c["apex_dist"], c["apex_speed"], color=APEX_COL, s=45, zorder=6)
        ax_speed.annotate(
            str(c["num"]), (c["apex_dist"], c["apex_speed"]),
            textcoords="offset points", xytext=(0, -13),
            fontsize=7, ha="center", color=APEX_LBL_COL, fontweight="bold",
        )

    ax_speed.set_ylabel("Speed (km/h)", color=TEXT_COL, fontsize=10)
    ax_speed.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT_COL)
    ax_speed.set_title(f"Driver vs Ideal Comparison: {driver_name}", color=TEXT_COL, fontsize=12, pad=10)
    plt.setp(ax_speed.get_xticklabels(), visible=False)

    # ── 2. Time Delta Trace ───────────────────────────────────────────────────
    ax_delta.plot(distance_grid, delta_trace, color="#ff4444", linewidth=1.5, label="Time Delta (s)")
    ax_delta.axhline(0, color=TEXT_COL, linewidth=1, linestyle="--", alpha=0.5)
    
    _add_sector_shading(ax_delta, sector_bounds)
    _add_sector_dividers(ax_delta, s1_end_dist, s2_end_dist)
    _add_corner_shading(ax_delta, corners)
    
    ax_delta.set_ylabel("Delta (s)", color=TEXT_COL, fontsize=10)
    ax_delta.set_xlabel("Distance (m)", color=TEXT_COL, fontsize=10)
    ax_delta.legend(loc="upper right", facecolor="#1a1a1a", edgecolor="#333333", labelcolor=TEXT_COL)

    # ── 3. Corner Analysis Table ──────────────────────────────────────────────
    ax_table.set_facecolor("#0a0a0a")
    ax_table.axis("off")
    
    headers = ["Corner", "Time Loss (s)", "Brake Dev (m)", "Throttle Dev (m)"]
    
    # Table layout parameters
    cols_x = [0.1, 0.3, 0.5, 0.7]
    y_start = 0.85
    row_h = 0.08
    
    # Draw headers
    ax_table.axhspan(y_start - row_h/2, y_start + row_h/2, color="#222233", alpha=0.8)
    for ci, h in enumerate(headers):
        ax_table.text(cols_x[ci], y_start, h, color="#aaaaff", fontweight="bold", ha="center", va="center")
    
    # Limit table rows to avoid overlap (max 10 rows per column, wrap if needed)
    y_pos = y_start - row_h
    x_offset = 0
    
    for i, res in enumerate(corner_analysis):
        if y_pos < 0.1: # start new column layout if it gets too low
            y_pos = y_start - row_h
            x_offset += 0.5
            if x_offset > 0.5: break # Only show up to 2 columns of table
            
        bg_col = "#1a1a2a" if i % 2 == 0 else "#0f1a0f"
        ax_table.axhspan(y_pos - row_h/2, y_pos + row_h/2, xmin=x_offset, xmax=x_offset+0.45, color=bg_col, alpha=0.5)
        
        c_num = str(res["num"])
        t_loss = f"{res['time_loss']:.3f}" if res['time_loss'] is not np.nan else "N/A"
        b_dev = f"{res['brake_deviation']:.1f}" if not np.isnan(res['brake_deviation']) else "N/A"
        t_dev = f"{res['throttle_deviation']:.1f}" if not np.isnan(res['throttle_deviation']) else "N/A"
        
        # Color coding time loss
        t_color = "#ff8888" if res['time_loss'] > 0.05 else ("#88ff88" if res['time_loss'] < -0.05 else TEXT_COL)
        
        ax_table.text(cols_x[0]/2 + x_offset, y_pos, f"T{c_num}", color=TEXT_COL, fontweight="bold", ha="center", va="center")
        ax_table.text(cols_x[1]/2 + x_offset + 0.1, y_pos, t_loss, color=t_color, ha="center", va="center")
        ax_table.text(cols_x[2]/2 + x_offset + 0.2, y_pos, b_dev, color=TEXT_COL, ha="center", va="center")
        ax_table.text(cols_x[3]/2 + x_offset + 0.3, y_pos, t_dev, color=TEXT_COL, ha="center", va="center")
        
        y_pos -= row_h

    ax_table.text(0.5, 0.98, "Corner-by-Corner Performance Breakdown", 
                  transform=ax_table.transAxes, fontsize=11, color="#aaaaaa", ha="center", va="top", fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved comparison plot → {save_path}")
    return fig
