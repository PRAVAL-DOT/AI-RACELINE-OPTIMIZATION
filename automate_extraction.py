import os
import logging
import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Any, Optional

# Setup clean production-grade logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GlobalF1Extractor")

# Enable FastF1 caching to save disk and prevent API rate-limiting
CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


class F1FeatureExtractor:
    def __init__(self, target_points: int = 3000, margin_pct: float = 1.03):
        """
        Args:
            target_points: Standard fixed distance grid resolution per lap.
            margin_pct: Threshold factor to screen pace laps (1.03 means max 103% of session minimum).
        """
        self.target_points = target_points
        self.margin_pct = margin_pct

    def extract_session_features(
        self, year: int, track: str, drivers_list: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Dynamically finds, cleans, and extracts feature tensors from a single GP weekend
        following the priority rule: Qualifying > Race > FP3 > FP2 > FP1.

        Returns:
            Tuple containing:
                - np.ndarray: A tensor of shape (N, target_points, 11)
                - List[Dict]: Metadata list for each extracted lap to guarantee synchronization.
        """
        session_priority = ["Q", "R", "FP3", "FP2", "FP1"]
        session = None
        chosen_type = None

        for s_type in session_priority:
            try:
                logger.info(f"Attempting to load {year} {track} - Session Type: {s_type}")
                session = fastf1.get_session(year, track, s_type)
                session.load(telemetry=True, weather=False)
                chosen_type = s_type
                break  
            except Exception as e:
                logger.warning(f"Session {s_type} not available for {track} ({e}). Trying next tier.")
                continue

        if session is None:
            raise ValueError(f"Could not load any valid session for {year} {track}.")

        logger.info(f"Successfully locked into [{chosen_type}] session for data extraction.")

        all_laps = session.laps
        if drivers_list:
            all_laps = all_laps.pick_drivers(drivers_list)

        clean_laps = all_laps.pick_quicklaps().dropna(subset=["LapTime"])

        if len(clean_laps) == 0:
            logger.error(f"No clean baseline laps found in {track} [{chosen_type}].")
            return np.empty((0, self.target_points, 11)), []

        clean_laps = clean_laps.copy()
        clean_laps["LapTimeSeconds"] = clean_laps["LapTime"].dt.total_seconds()
        
        best_session_time = clean_laps["LapTimeSeconds"].min()
        max_pace_cutoff = best_session_time * self.margin_pct
        
        pace_laps = clean_laps[clean_laps["LapTimeSeconds"] <= max_pace_cutoff]
        logger.info(f"Pace Filtering: Selected {len(pace_laps)} out of {len(clean_laps)} laps (Cutoff: {max_pace_cutoff:.2f}s).")

        if len(pace_laps) == 0:
            return np.empty((0, self.target_points, 11)), []

        fastest_lap = pace_laps.sort_values(by="LapTimeSeconds").iloc[0]
        ref_tel = fastest_lap.get_telemetry().add_distance()
        max_track_distance = ref_tel["Distance"].max()
        
        target_distance_grid = np.linspace(0, max_track_distance, self.target_points)
        lap_tensors = []
        lap_metadata_records = []

        for idx, lap in pace_laps.iterrows():
            try:
                tel = lap.get_telemetry().add_distance()
                
                if len(tel) < 50 or tel["Distance"].max() < (max_track_distance * 0.98):
                    continue 

                src_dist = tel["Distance"].values
                src_speed = tel["Speed"].values
                src_throttle = tel["Throttle"].values
                src_brake = tel["Brake"].values.astype(float) 

                f_speed = interp1d(src_dist, src_speed, kind="linear", fill_value="extrapolate")
                f_throttle = interp1d(src_dist, src_throttle, kind="linear", fill_value="extrapolate")
                f_brake = interp1d(src_dist, src_brake, kind="linear", fill_value="extrapolate")

                speed = f_speed(target_distance_grid)
                throttle = np.clip(f_throttle(target_distance_grid), 0, 100)
                brake = np.clip(f_brake(target_distance_grid), 0, 1)

                dx = target_distance_grid[1] - target_distance_grid[0]
                
                speed_gradient = np.gradient(speed, dx)
                throttle_gradient = np.gradient(throttle, dx)
                brake_gradient = np.gradient(brake, dx)

                speed_mps = speed / 3.6  
                acceleration = speed_mps * speed_gradient  
                jerk = speed_mps * np.gradient(acceleration, dx)  

                normalized_distance = target_distance_grid / max_track_distance

                corner_id = np.full_like(target_distance_grid, -1.0)
                corner_phase = np.zeros_like(target_distance_grid)

                lap_features = np.column_stack([
                    normalized_distance, # 0
                    speed,               # 1
                    throttle,            # 2
                    brake,               # 3
                    acceleration,        # 4
                    jerk,                # 5
                    speed_gradient,      # 6
                    throttle_gradient,   # 7
                    brake_gradient,      # 8
                    corner_id,           # 9
                    corner_phase         # 10
                ])

                lap_tensors.append(lap_features)
                
                # Capture synchronized raw metadata indicators per successful lap array
                lap_metadata_records.append({
                    "Driver": str(lap["Driver"]),
                    "DriverNumber": str(lap["DriverNumber"]),
                    "Track": track,
                    "Session": chosen_type,
                    "LapTime": lap["LapTimeSeconds"]
                })

            except Exception as e:
                logger.debug(f"Skipping lap index {idx} due to calculation anomaly: {e}")
                continue

        if not lap_tensors:
            return np.empty((0, self.target_points, 11)), []

        session_tensor = np.stack(lap_tensors, axis=0)
        return session_tensor, lap_metadata_records


def run_automated_pipeline():
    # Define extraction parameters explicitly targetting the 'data' directory
    YEAR = 2025
    TRACKS_TO_EXTRACT = ["Bahrain", "Suzuka", "Silverstone", "Monaco", "Monza"]
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)

    extractor = F1FeatureExtractor(target_points=3000, margin_pct=1.03)
    
    global_tensors: List[np.ndarray] = []
    global_metadata_raw: List[Dict[str, Any]] = []
    
    # Track global drivers discovered dynamically across the pipeline run to construct maps from scratch
    discovered_drivers = set()

    print("=" * 60)
    print(f"      STARTING AUTOMATED MULTI-RACE DATASET GENERATION        ")
    print("=" * 60)

    for track in TRACKS_TO_EXTRACT:
        print(f"\n🚀 Processing Track: {track.upper()}...")
        try:
            # Synchronously extract features and internal structural records
            track_tensor, track_meta = extractor.extract_session_features(year=YEAR, track=track)
            
            if track_tensor.shape[0] > 0:
                logger.info(f"Successfully extracted {track_tensor.shape[0]} laps for {track}.")
                global_tensors.append(track_tensor)
                global_metadata_raw.extend(track_meta)
                
                # Update globally discovered driver tracking
                for meta in track_meta:
                    discovered_drivers.add((meta["Driver"], meta["DriverNumber"]))
                
                # Save track-specific arrays inside data/ folder
                track_output_path = os.path.join(DATA_DIR, f"{track}_{YEAR}_features.npy")
                np.save(track_output_path, track_tensor)
                print(f"  ↳ Saved individual circuit matrix to {track_output_path}")
            else:
                logger.warning(f"No pace laps matched filtering for {track}. Skipping tracking records.")

        except Exception as e:
            logger.error(f"Failed processing {track} entirely due to an unhandled error: {e}")
            print("Moving to next track in list...")
            continue

    # --- Global Mapping & Consolidation Engine ---
    print("\n" + "=" * 60)
    print("      CONSOLIDATING COMPYLED TELEMETRY DATASETS & LABELS      ")
    print("=" * 60)

    if global_tensors and global_metadata_raw:
        # 1. Finalize Deterministic Label Encodings from Scratch (Ordered Alphabetically by Driver Code)
        sorted_drivers = sorted(list(discovered_drivers), key=lambda x: x[0])
        
        driver_to_label: Dict[str, int] = {item[0]: idx for idx, item in enumerate(sorted_drivers)}
        
        # Save label_mapping.csv
        mapping_rows = [{"Driver": d[0], "DriverNumber": d[1], "Label": driver_to_label[d[0]]} for d in sorted_drivers]
        df_mapping = pd.DataFrame(mapping_rows)
        mapping_output_path = os.path.join(DATA_DIR, "label_mapping.csv")
        df_mapping.to_csv(mapping_output_path, index=False)
        print(f"📁 Saved Label Mappings to: {mapping_output_path}")

        # 2. Process Categorization Labels and Metadata Indices Synchronously
        final_labels: List[int] = []
        final_metadata_rows: List[Dict[str, Any]] = []
        
        for lap_idx, meta in enumerate(global_metadata_raw):
            lbl = driver_to_label[meta["Driver"]]
            final_labels.append(lbl)
            
            # Combine raw indicators with synchronized structural labels and array index reference
            final_metadata_rows.append({
                "Driver": meta["Driver"],
                "DriverNumber": meta["DriverNumber"],
                "Label": lbl,
                "Track": meta["Track"],
                "Session": meta["Session"],
                "LapTime": meta["LapTime"],
                "LapIndex": lap_idx
            })

        # 3. Stack and Save 3D Feature Matrix Tensors
        master_dataset = np.concatenate(global_tensors, axis=0)
        master_output_path = os.path.join(DATA_DIR, f"master_dataset_{YEAR}_laps.npy")
        np.save(master_output_path, master_dataset)
        
        # 4. Save Vectorized Integer Target Array
        labels_array = np.array(final_labels, dtype=np.int32)
        labels_output_path = os.path.join(DATA_DIR, "driver_labels.npy")
        np.save(labels_output_path, labels_array)
        
        # 5. Save Structured Metadata Reference Frame
        df_metadata = pd.DataFrame(final_metadata_rows)
        metadata_output_path = os.path.join(DATA_DIR, "metadata.csv")
        df_metadata.to_csv(metadata_output_path, index=False)

        print(f"\n🎉 SUCCESS! Entire pipeline dataset rebuilt cleanly from scratch.")
        print(f"📁 Master Feature Tensor Saved to: {master_output_path} -> Structural Shape: {master_dataset.shape}")
        print(f"📁 Vector Labels Saved to        : {labels_output_path} -> Array Shape: {labels_array.shape}")
        print(f"📁 Track Synchronized CSV Saved to: {metadata_output_path} -> Dimensions: {df_metadata.shape}")
        
        # Guardrail Validation Test Verification
        assert master_dataset.shape[0] == labels_array.shape[0] == len(df_metadata), \
            "Critical Alignment Exception: Synchronization mismatch across saved pipeline file dimensions."
        logger.info("Pipeline Alignment Check Passed: Array shapes and tracking indexes match flawlessly.")
        
    else:
        logger.error("No valid telemetry structures could be extracted from any listed circuits.")


if __name__ == "__main__":
    run_automated_pipeline()