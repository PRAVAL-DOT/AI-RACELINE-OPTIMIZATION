"""
tests/test_ideal_profile.py
---------------------------
Unit tests for ideal_profile.py — run with:
    pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ideal_profile import (
    build_ideal_profile,
    estimate_sector_time,
    compute_lap_times,
    fmt_time,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_resampled(n: int = 500, n_laps: int = 5) -> dict:
    """Synthetic resampled data for testing."""
    rng = np.random.default_rng(42)
    return {
        "distance_grid": np.linspace(0, 5400, n),
        "speed":         rng.uniform(80, 330, (n_laps, n)),
        "throttle":      rng.uniform(0,   1,  (n_laps, n)),
        "brake":         rng.uniform(0,   1,  (n_laps, n)),
    }


# ── build_ideal_profile ───────────────────────────────────────────────────────

def test_ideal_profile_shape():
    data = _make_resampled(n=500)
    profile = build_ideal_profile(data)
    assert profile["speed"].shape    == (500,)
    assert profile["throttle"].shape == (500,)
    assert profile["brake"].shape    == (500,)


def test_ideal_speed_is_90th_percentile_order():
    """Ideal speed should be >= median speed at most points."""
    data    = _make_resampled(n=300)
    profile = build_ideal_profile(data, smooth_sigma=0)
    median  = np.median(data["speed"], axis=0)
    assert np.mean(profile["speed"] >= median) > 0.85


# ── estimate_sector_time ──────────────────────────────────────────────────────

def test_sector_time_constant_speed():
    """At constant 180 km/h over 1800m, time should be exactly 36s."""
    dist  = np.linspace(0, 1800, 1000)
    speed = np.full(1000, 180.0)
    t     = estimate_sector_time(speed, dist)
    assert abs(t - 36.0) < 0.1, f"Expected ~36s, got {t:.3f}s"


def test_sector_time_positive():
    dist  = np.linspace(0, 5400, 3000)
    speed = np.random.uniform(80, 330, 3000)
    t     = estimate_sector_time(speed, dist)
    assert t > 0


# ── compute_lap_times ─────────────────────────────────────────────────────────

def test_lap_time_equals_sector_sum():
    data    = _make_resampled(n=3000)
    grid    = data["distance_grid"]
    profile = build_ideal_profile(data)
    times   = compute_lap_times(profile, grid, s1_end_idx=1000, s2_end_idx=2000)
    assert abs(times["s1"] + times["s2"] + times["s3"] - times["lap"]) < 1e-6


# ── fmt_time ──────────────────────────────────────────────────────────────────

def test_fmt_time_known_values():
    assert fmt_time(90.0)   == "1:30.000"
    assert fmt_time(0.0)    == "0:00.000"
    assert fmt_time(61.5)   == "1:01.500"
    assert fmt_time(3661.0) == "61:01.000"
