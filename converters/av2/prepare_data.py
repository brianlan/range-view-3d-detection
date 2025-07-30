"""Tool export the Argoverse 2 dataset with range images."""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

# Fix NumPy compatibility issues
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex

# Add src directory to path for torchbox3d imports
script_dir = Path(__file__).parent.parent.parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

import polars as pl
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.utils.io import read_city_SE3_ego
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from utils import build_range_view, correct_laser_numbers, unmotion_compensate

# Constants
TRANSLATION: Final = ("tx_m", "ty_m", "tz_m")
DIMS: Final = ("length_m", "width_m", "height_m")
QUATERNION_WXYZ: Final = ("qx", "qy", "qz", "qw")
UP_LIDAR_MAX_LASER: Final = 31
DOWN_LIDAR_MIN_LASER: Final = 32
DOWN_LIDAR_OFFSET: Final = 32

FEATURE_COLUMN_NAMES: Tuple[str, ...] = (
    "x",
    "y",
    "z",
    "intensity",
    "laser_number",
    "is_within_roi",
)


def validate_inputs(src_root_dir: Path, dst_root_dir: Path, splits: List[str]) -> None:
    """Validate input directories and required files exist.
    
    Args:
        src_root_dir: Source root directory path
        dst_root_dir: Destination root directory path  
        splits: List of data splits to validate
        
    Raises:
        FileNotFoundError: If required directories or files don't exist
    """
    if not src_root_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_root_dir}")
    
    for split in splits:
        split_dir = src_root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    
    # Create destination directory if it doesn't exist
    dst_root_dir.mkdir(parents=True, exist_ok=True)


def export_dataset(
    src_root_dir: Path,
    dst_root_dir: Path,
    range_view_config: Dict[str, Any],
    enable_write: bool = False,
) -> None:
    """Export the Argoverse 2 dataset with range images.
    
    Args:
        src_root_dir: Root directory for the Argoverse 2 sensor dataset
        dst_root_dir: Destination directory for the processed data
        range_view_config: Configuration dictionary for range view processing
        enable_write: Whether to write output files (False for dry run)
        
    Raises:
        FileNotFoundError: If required input files are missing
        KeyError: If required configuration keys are missing
    """
    # Extract configuration with error handling
    try:
        build_uniform_inclination = range_view_config["build_uniform_inclination"]
        sensor_name = range_view_config["sensor_name"]
        height = range_view_config["height"]
        width = range_view_config["width"]
        export_range_view = range_view_config["export_range_view"]
        enable_motion_uncompensation = range_view_config["enable_motion_uncompensation"]
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}") from e

    splits = ["train", "val"]
    validate_inputs(src_root_dir, dst_root_dir, splits)
    
    for split in splits:
        split_dir = src_root_dir / split
        log_dirs = sorted(split_dir.glob("*"))
        
        for log_dir in tqdm(log_dirs, desc=f"Processing {split} split"):
            try:
                avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)
            except Exception as e:
                logging.warning(f"Failed to load map for {log_dir}: {e}")
                continue

            log_id = log_dir.stem
            dst_log_dir = dst_root_dir / split / log_id
            dst_log_dir.mkdir(exist_ok=True, parents=True)

            # Load annotations
            annotations_path = log_dir / "annotations.feather"
            if not annotations_path.exists():
                logging.warning(f"Annotations file not found: {annotations_path}")
                continue
                
            annotations = (
                pl.scan_ipc(annotations_path)
                .filter(pl.col("num_interior_pts") > 0)
                .collect()
            )

            # Load poses and setup interpolation
            poses_path = log_dir / "city_SE3_egovehicle.feather"
            if not poses_path.exists():
                logging.warning(f"Poses file not found: {poses_path}")
                continue
                
            poses = pl.read_ipc(poses_path)
            rots = Rotation.from_quat(poses.select(QUATERNION_WXYZ).to_numpy())

            times = poses["timestamp_ns"].to_numpy()
            slerp = Slerp(times, rots)

            # Load extrinsics
            extrinsics_path = log_dir / "calibration" / "egovehicle_SE3_sensor.feather"
            if not extrinsics_path.exists():
                logging.warning(f"Extrinsics file not found: {extrinsics_path}")
                continue
                
            extrinsics = pl.read_ipc(extrinsics_path, memory_map=False)

            try:
                city_SE3_egovehicle = read_city_SE3_ego(log_dir)
            except Exception as e:
                logging.warning(f"Failed to read city SE3 ego for {log_dir}: {e}")
                continue

            sweep_paths = sorted((log_dir / "sensors" / "lidar").glob("*.feather"))
            for sweep_path in sweep_paths:
                try:
                    timestamp_ns = int(sweep_path.stem)
                    lidar_lazy = (
                        pl.scan_ipc(sweep_path)
                        .select(("x", "y", "z", "intensity", "laser_number", "offset_ns"))
                        .cast({"x": pl.Float64, "y": pl.Float64, "z": pl.Float64})
                    )

                    # Filter based on sensor name
                    if sensor_name == "up_lidar":
                        lidar_lazy = lidar_lazy.filter(pl.col("laser_number") <= UP_LIDAR_MAX_LASER)
                    elif sensor_name == "down_lidar":
                        lidar_lazy = lidar_lazy.filter(
                            pl.col("laser_number") >= DOWN_LIDAR_MIN_LASER
                        ).with_columns(laser_number=pl.col("laser_number") - DOWN_LIDAR_OFFSET)
                    
                    lidar = lidar_lazy.collect()
                except Exception as e:
                    logging.warning(f"Failed to process sweep {sweep_path}: {e}")
                    continue

                city_xyz = city_SE3_egovehicle[timestamp_ns].transform_from(
                    lidar.select(("x", "y", "z")).to_numpy()
                )
                mask = avm.get_raster_layer_points_boolean(
                    city_xyz, RasterLayerType.ROI
                )
                lidar = lidar.with_columns(is_within_roi=pl.lit(mask))

                if enable_motion_uncompensation:
                    lidar = unmotion_compensate(
                        lidar,
                        poses,
                        timestamp_ns,
                        slerp,
                    )
                else:
                    lidar = lidar.with_columns(
                        x_p=pl.col("x"), y_p=pl.col("y"), z_p=pl.col("z")
                    )

                features = lidar.select(FEATURE_COLUMN_NAMES)
                laser_number = lidar["laser_number"].to_numpy().copy()

                laser_number = correct_laser_numbers(
                    laser_number,
                    log_id,
                    height=height,
                )

                if sensor_name == "down_lidar":
                    laser_number = DOWN_LIDAR_OFFSET - laser_number - 1

                lidar = lidar.with_columns(pl.Series("laser_number", laser_number))
                
                if export_range_view:
                    range_view = build_range_view(
                        lidar,
                        extrinsics=extrinsics,
                        features=features.to_numpy(),
                        sensor_name=sensor_name if sensor_name != "all" else "up_lidar",
                        height=height,
                        width=width,
                        build_uniform_inclination=build_uniform_inclination,
                    )

                    if enable_write:
                        range_view_dst = (
                            dst_log_dir
                            / "sensors"
                            / "range_view"
                            / f"{timestamp_ns}.feather"
                        )
                        range_view_dst.parent.mkdir(parents=True, exist_ok=True)
                        range_view.write_ipc(range_view_dst)

                if enable_write:
                    lidar_dst = (
                        dst_log_dir / "sensors" / "lidar" / f"{timestamp_ns}.feather"
                    )
                    lidar_dst.parent.mkdir(parents=True, exist_ok=True)
                    lidar.write_ipc(lidar_dst)

            if enable_write:
                try:
                    annotations_dst = dst_log_dir / "annotations.feather"
                    annotations.write_ipc(annotations_dst)

                    poses_dst = dst_log_dir / "city_SE3_egovehicle.feather"
                    poses.write_ipc(poses_dst)

                    shutil.copytree(
                        log_dir / "map", dst_log_dir / "map", dirs_exist_ok=True
                    )
                except Exception as e:
                    logging.error(f"Failed to write output files for {log_dir}: {e}")
                    continue


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export the Argoverse 2 dataset with range images."
    )
    
    # Required arguments
    parser.add_argument(
        "--src-root-dir",
        type=Path,
        required=True,
        help="Root directory for the Argoverse 2 sensor dataset (raw files)"
    )
    parser.add_argument(
        "--dst-root-dir", 
        type=Path,
        required=True,
        help="Destination directory for the processed data"
    )
    
    # Range view configuration arguments
    parser.add_argument(
        "--sensor-name",
        type=str,
        default="up_lidar",
        choices=["up_lidar", "down_lidar", "all"],
        help="Sensor name to process (default: up_lidar)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Height of the range view (default: 32)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1800,
        help="Width of the range view (default: 1800)"
    )
    parser.add_argument(
        "--build-uniform-inclination",
        action="store_true",
        help="Build uniform inclination for range view"
    )
    parser.add_argument(
        "--no-export-range-view",
        action="store_true",
        help="Disable range view export"
    )
    parser.add_argument(
        "--disable-motion-uncompensation",
        action="store_true",
        help="Disable motion uncompensation"
    )
    
    # Control arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing files (dry run mode)"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        args = parse_args()
        
        # Build range view configuration from arguments
        range_view_config = {
            "build_uniform_inclination": args.build_uniform_inclination,
            "sensor_name": args.sensor_name,
            "height": args.height,
            "width": args.width,
            "export_range_view": not args.no_export_range_view,
            "enable_motion_uncompensation": not args.disable_motion_uncompensation,
        }
        
        # Enable writing unless it's a dry run
        enable_write = not args.dry_run
        
        logging.info(f"Source directory: {args.src_root_dir}")
        logging.info(f"Destination directory: {args.dst_root_dir}")
        logging.info(f"Range view config: {range_view_config}")
        logging.info(f"Enable write: {enable_write}")
        
        export_dataset(
            src_root_dir=args.src_root_dir,
            dst_root_dir=args.dst_root_dir,
            range_view_config=range_view_config,
            enable_write=enable_write,
        )
        
        logging.info("Export completed successfully")
        
    except Exception as e:
        logging.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
