# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025 (Phase 1 complete)

### Added
- FastF1 session loading with local disk cache
- Distance-normalised telemetry resampling (3000-point grid, ~1.8m resolution)
- Ideal profile construction using 90th-percentile speed/throttle aggregation
- Official corner detection via `session.get_circuit_info().corners`
  — replaces unreliable speed-threshold heuristics; correctly resolves all 15
  Bahrain corners including chicanes
- Exact sector split distances derived from fastest lap telemetry timestamps
- Kinematic lap time estimation (dt = ds/v integration) per sector
- 4-panel dark-theme dashboard: speed + throttle + brake + reference time table
- CLI interface (`main.py`) with `--year`, `--gp`, `--session`, `--output` flags
- Modular src/ layout: `telemetry.py`, `ideal_profile.py`, `visualization.py`
- Unit tests for core numerical functions (pytest)
- `.gitignore` excluding cache/ and outputs/

### Fixed
- Distance axis showing ~50,000m instead of ~5,400m (was computing from raw
  X/Y coordinates; now uses FastF1's built-in Distance column)
- Corner detection finding 9 instead of 15 corners (replaced percentile
  threshold with official circuit geometry anchor points)
- Sector boundaries hardcoded; now derived dynamically from lap telemetry

---

## [Unreleased] — Phase 2 (planned)

- Per-driver vs ideal delta plot (time loss per corner)
- Braking point deviation analysis
- Throttle application point comparison
- Corner-by-corner performance breakdown table
