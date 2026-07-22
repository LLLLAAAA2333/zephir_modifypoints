"""Create a native ZephIR 1.0.5 dataset from TIFF frames and dynamics.h5.

The output directory contains:

* data.h5:        uint8 image data with shape (T, 1, Z, Y, X)
* annotations.h5: native ZephIR annotation fields (no ``abs_t_idx``)
* worldlines.h5:  one row per neuron/worldline
* metadata.json:  ZephIR dimensions plus source-volume provenance

Only the volumes listed in ``SELECTED_VOLUMES`` are exported. Their source
volume numbers are remapped to consecutive ZephIR time indices 0..T-1.
"""

from __future__ import annotations

import colorsys
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import tifffile
from scipy.ndimage import affine_transform
from tqdm import tqdm


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# Reference/red TIFF source. A directory should contain numbered frames such
# as 00000000.tif. A TIFF file is treated as a multi-page stack.
TIFF_PATH = Path(r"\\192.168.1.192\Ikrma-2\20260304\W3_2026-03-05_01-03-23\0_Camera-Red_VSC-10629")
DYNAMICS_PATH = Path(r"Z:\data5\CBMI_inferred_results\Ikrma\proxy\20260304_w3\dynamics.h5")
OUTPUT_DIR = Path(r"H:\Process_temporary\WJH\zephir_modifypoints\data\20260304_w3")

# Source volume numbers, in the desired output order. ZephIR t_idx is remapped
# to 0, 1, 2, ... while these source numbers are retained in metadata.json.
SELECTED_VOLUMES = [551, 571, 594]

# TIFF acquisition layout. Z_START_FRAME and Z_END_FRAME are inclusive offsets
# inside each complete volume.
FRAMES_PER_VOLUME = 20
Z_START_FRAME = 1
Z_END_FRAME = 17
REVERSE_Z_BY_VOLUME_PARITY = (False, False)

# One intensity transform is calibrated from the first selected volume, then
# reused for every selected volume to preserve temporal contrast.
INTENSITY_MIN = 110
INTENSITY_PERCENTILE = 99.0
# Set an explicit value to avoid data-dependent calibration, e.g. 900.
INTENSITY_MAX = None

# dynamics.h5 contains one group per source volume. Numeric group names such as
# 000000 are used as source volume IDs; non-numeric names fall back to position.
DYNAMICS_FIRST_VOLUME = 0
POINT_DATASET_CANDIDATES = (
    "d_neuron_pt_tuple_matched_raw_vol",
    "neuron_pt_tuple",
)
CENTER_DATASET = "center"
ROTATION_DATASET = "rot"

# Apply the volume-specific PCA body-pose correction to both the image and
# points. PCA +Y is downward in image coordinates. An additional +90-degree
# row-vector rotation maps +Y to -X, so the worm points left in the annotator.
ALIGN_XY = True
GOAL_ANGLE_DEGREES = 90.0
# Mirror the final aligned image and points horizontally when True.
FLIP_X = True
# Mirror the final aligned image and points vertically when True.
FLIP_Y = False
IMAGE_INTERPOLATION_ORDER = 1

# Common XY crop applied after alignment and flips. Options:
#   "none":        keep the full TIFF canvas
#   "point_cloud": union of selected-volume point bounds plus CROP_PADDING
#   "custom":      use CUSTOM_CROP_XYXY = (x0, y0, x1, y1), with x1/y1 exclusive
CROP_MODE = "point_cloud"
CROP_PADDING = 20.0
CUSTOM_CROP_XYXY = None

# ``d_neuron_pt_tuple_matched_raw_vol`` stores X,Y,Z in columns 0,1,2.
# Z is expressed in XY-pixel units (slice_index * Z_STEP_SIZE / XY_PIXEL_SIZE).
COORDINATE_ORDER = "xyz"
XY_PIXEL_SIZE = 0.3
Z_STEP_SIZE = 1.5

PROVENANCE = b"ANTT"
H5_COMPRESSION = None  # Set to "gzip" to reduce size at the cost of speed.
OVERWRITE = False


def _natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


class TiffFrameSource:
    """Read numbered TIFF files or individual pages of a TIFF stack."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stack: tifffile.TiffFile | None = None
        self._frame_paths: dict[int, Path] = {}
        self._directory_indexed = False

    def __enter__(self) -> "TiffFrameSource":
        if self.path.is_dir():
            # Numeric eight-digit names are accessed directly in read(). This
            # avoids an expensive full-directory scan on network shares.
            pass
        elif self.path.is_file():
            self._stack = tifffile.TiffFile(self.path)
        else:
            raise FileNotFoundError(f"TIFF source does not exist: {self.path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()

    def _index_directory(self) -> None:
        if self._directory_indexed:
            return
        for candidate in self.path.iterdir():
            if candidate.suffix.lower() not in {".tif", ".tiff"}:
                continue
            try:
                frame_number = int(candidate.stem)
            except ValueError:
                continue
            if frame_number in self._frame_paths:
                raise ValueError(
                    f"Duplicate TIFF frame number {frame_number} in {self.path}"
                )
            self._frame_paths[frame_number] = candidate
        self._directory_indexed = True
        if not self._frame_paths:
            raise ValueError(f"No numerically named TIFF frames found in {self.path}")

    def read(self, frame_number: int) -> np.ndarray:
        if self._stack is not None:
            if not 0 <= frame_number < len(self._stack.pages):
                raise IndexError(
                    f"TIFF page {frame_number} is outside [0, {len(self._stack.pages)}) "
                    f"for {self.path}"
                )
            frame = self._stack.pages[frame_number].asarray()
        else:
            direct_candidates = (
                self.path / f"{frame_number:08d}.tif",
                self.path / f"{frame_number:08d}.tiff",
                self.path / f"{frame_number}.tif",
                self.path / f"{frame_number}.tiff",
            )
            frame_path = next(
                (candidate for candidate in direct_candidates if candidate.is_file()),
                None,
            )
            if frame_path is None:
                self._index_directory()
                try:
                    frame_path = self._frame_paths[frame_number]
                except KeyError as error:
                    raise FileNotFoundError(
                        f"Missing TIFF frame {frame_number:08d} in {self.path}"
                    ) from error
            frame = tifffile.imread(frame_path)

        frame = np.asarray(frame)
        if frame.ndim != 2:
            raise ValueError(
                f"Expected a 2-D TIFF frame, got shape {frame.shape} at frame "
                f"{frame_number} in {self.path}"
            )
        return frame


def _validate_settings(
    tiff_path: Path,
    selected_volumes: Sequence[int],
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z_by_volume_parity: Sequence[bool],
    coordinate_order: str,
    interpolation_order: int,
) -> None:
    if not Path(tiff_path).exists():
        raise FileNotFoundError(f"TIFF source does not exist: {tiff_path}")
    if not selected_volumes:
        raise ValueError("SELECTED_VOLUMES must not be empty")
    if len(set(selected_volumes)) != len(selected_volumes):
        raise ValueError("SELECTED_VOLUMES contains duplicates")
    if any(int(volume) != volume or volume < 0 for volume in selected_volumes):
        raise ValueError("Selected volume numbers must be non-negative integers")
    if frames_per_volume <= 0:
        raise ValueError("FRAMES_PER_VOLUME must be positive")
    if not 0 <= z_start_frame <= z_end_frame < frames_per_volume:
        raise ValueError(
            "Z frame range must satisfy 0 <= start <= end < FRAMES_PER_VOLUME"
        )
    if len(reverse_z_by_volume_parity) != 2:
        raise ValueError("REVERSE_Z_BY_VOLUME_PARITY must contain two booleans")
    if len(coordinate_order) != 3 or set(coordinate_order.lower()) != set("xyz"):
        raise ValueError("COORDINATE_ORDER must be a permutation of 'xyz'")
    if interpolation_order not in (0, 1, 3):
        raise ValueError("IMAGE_INTERPOLATION_ORDER must be 0, 1, or 3")


def _read_volume(
    source: TiffFrameSource,
    volume_number: int,
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z_by_volume_parity: Sequence[bool],
) -> np.ndarray:
    frame_numbers = list(
        range(
            volume_number * frames_per_volume + z_start_frame,
            volume_number * frames_per_volume + z_end_frame + 1,
        )
    )
    if reverse_z_by_volume_parity[volume_number % 2]:
        frame_numbers.reverse()

    frames = [source.read(frame_number) for frame_number in frame_numbers]
    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames):
        raise ValueError(
            f"Inconsistent TIFF shapes within source volume {volume_number}"
        )

    return np.stack(frames, axis=0)


def _read_volume_shape(
    tiff_path: Path,
    volume_number: int,
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
) -> tuple[int, int, int]:
    """Read one TIFF plane and return the expected full (Z,Y,X) shape."""

    frame_number = volume_number * frames_per_volume + z_start_frame
    with TiffFrameSource(Path(tiff_path)) as source:
        frame = source.read(frame_number)
    return (z_end_frame - z_start_frame + 1, int(frame.shape[0]), int(frame.shape[1]))


def _calibrate_intensity_max(
    volume: np.ndarray,
    intensity_min: float,
    percentile: float,
) -> float:
    foreground = volume[volume > intensity_min]
    if foreground.size == 0:
        raise ValueError(
            f"No voxels exceed INTENSITY_MIN={intensity_min}; set a lower threshold"
        )
    value = float(np.percentile(foreground, percentile))
    if value <= intensity_min:
        raise ValueError(
            f"Calibrated intensity maximum {value} is not above {intensity_min}"
        )
    return value


def _to_uint8(volume: np.ndarray, intensity_min: float, intensity_max: float) -> np.ndarray:
    if intensity_max <= intensity_min:
        raise ValueError("Intensity maximum must be greater than intensity minimum")
    scaled = np.asarray(volume, dtype=np.float32)
    scaled = (scaled - intensity_min) * (255.0 / (intensity_max - intensity_min))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def image_xy_rotation_matrix(angle_degrees: float) -> np.ndarray:
    """Return a row-vector XY rotation matrix for image coordinates.

    Image X points right and Y points down. With row-vector multiplication,
    +90 degrees maps +Y (down) to -X (left).
    """

    angle_radians = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return np.asarray(
        [[cosine, sine], [-sine, cosine]], dtype=np.float32
    )


def _output_center_xy(image_shape_yx: Sequence[int]) -> np.ndarray:
    height, width = (int(value) for value in image_shape_yx)
    # ZephIR normalizes as x/width and y/height, so this maps the body center
    # exactly to normalized coordinate (0.5, 0.5).
    return np.asarray([width / 2.0, height / 2.0], dtype=np.float32)


def align_points_xy(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    image_shape_yx: Sequence[int],
) -> np.ndarray:
    """Map raw row-vector XY points to the centered PCA-aligned canvas."""

    output_center = _output_center_xy(image_shape_yx)
    return (
        (np.asarray(points_xy, dtype=np.float32) - center_xy) @ rotation_xy
        + output_center
    )


def flip_points_x(points_xy: np.ndarray, image_width: int) -> np.ndarray:
    """Mirror XY point coordinates exactly like image[..., ::-1]."""

    mirrored = np.asarray(points_xy, dtype=np.float32).copy()
    mirrored[:, 0] = (int(image_width) - 1) - mirrored[:, 0]
    return mirrored


def flip_points_y(points_xy: np.ndarray, image_height: int) -> np.ndarray:
    """Mirror XY point coordinates exactly like image[:, ::-1, :]."""

    mirrored = np.asarray(points_xy, dtype=np.float32).copy()
    mirrored[:, 1] = (int(image_height) - 1) - mirrored[:, 1]
    return mirrored


def transform_points_xy(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    image_shape_yx: Sequence[int],
    align_xy: bool,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """Apply the same pre-crop XY transforms used for the image volume."""

    height, width = (int(value) for value in image_shape_yx)
    transformed = np.asarray(points_xy, dtype=np.float32).copy()
    if align_xy:
        transformed = align_points_xy(
            transformed, center_xy, rotation_xy, (height, width)
        )
    if flip_x:
        transformed = flip_points_x(transformed, width)
    if flip_y:
        transformed = flip_points_y(transformed, height)
    return transformed


def align_volume_xy(
    volume_zyx: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Apply the point-cloud XY transform to every Z slice of an image volume."""

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Volume must have shape (Z,Y,X), got {volume.shape}")

    # Point forward map (row vectors):
    #   aligned_xy = (raw_xy - center_xy) @ rotation_xy + output_center_xy
    # scipy.ndimage.affine_transform needs the inverse map in YX coordinates.
    swap_xy_yx = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    rotation = np.asarray(rotation_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    output_center = _output_center_xy(volume.shape[1:]).astype(np.float64)
    inverse_matrix_yx = swap_xy_yx @ rotation @ swap_xy_yx
    inverse_offset_yx = swap_xy_yx @ (center - rotation @ output_center)

    matrix_zyx = np.eye(3, dtype=np.float64)
    matrix_zyx[1:, 1:] = inverse_matrix_yx
    offset_zyx = np.asarray(
        [0.0, inverse_offset_yx[0], inverse_offset_yx[1]], dtype=np.float64
    )
    return affine_transform(
        np.asarray(volume, dtype=np.float32),
        matrix=matrix_zyx,
        offset=offset_zyx,
        output_shape=volume.shape,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
        prefilter=interpolation_order > 1,
    )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    temporary = _temporary_path(path)
    if temporary.exists():
        temporary.unlink()
    return temporary


def create_data_h5(
    tiff_path: Path,
    output_path: Path,
    selected_volumes: Sequence[int],
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z_by_volume_parity: Sequence[bool],
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
    align_xy: bool,
    flip_x: bool,
    flip_y: bool,
    crop_bounds_xyxy: Sequence[int],
    interpolation_order: int,
    intensity_min: float,
    intensity_percentile: float,
    intensity_max: float | None = None,
    compression: str | None = None,
    overwrite: bool = False,
) -> tuple[tuple[int, int, int, int, int], float]:
    """Write a single-channel data.h5 and return its shape/intensity maximum."""

    source = TiffFrameSource(Path(tiff_path))
    temporary: Path | None = None
    try:
        source.__enter__()
        temporary = _prepare_output(Path(output_path), overwrite)

        first_volume = _read_volume(
            source,
            int(selected_volumes[0]),
            frames_per_volume,
            z_start_frame,
            z_end_frame,
            reverse_z_by_volume_parity,
        )
        reference_shape = first_volume.shape
        x0, y0, x1, y1 = (int(value) for value in crop_bounds_xyxy)
        if not (
            0 <= x0 < x1 <= reference_shape[2]
            and 0 <= y0 < y1 <= reference_shape[1]
        ):
            raise ValueError(
                f"Crop {(x0, y0, x1, y1)} is invalid for volume shape "
                f"{reference_shape}"
            )
        cropped_height = y1 - y0
        cropped_width = x1 - x0

        if intensity_max is None:
            calibrated_max = _calibrate_intensity_max(
                first_volume, intensity_min, intensity_percentile
            )
        else:
            calibrated_max = float(intensity_max)

        shape = (
            len(selected_volumes),
            1,
            reference_shape[0],
            cropped_height,
            cropped_width,
        )
        chunks = (
            1,
            1,
            reference_shape[0],
            min(cropped_height, 256),
            min(cropped_width, 256),
        )

        with h5py.File(temporary, "w") as h5_file:
            data = h5_file.create_dataset(
                "data",
                shape=shape,
                dtype=np.uint8,
                chunks=chunks,
                compression=compression,
            )
            volume_iterator = tqdm(
                enumerate(selected_volumes),
                total=len(selected_volumes),
                desc="Writing aligned ZephIR volumes",
            )
            for local_t, source_volume in volume_iterator:
                volume = (
                    first_volume
                    if local_t == 0
                    else _read_volume(
                        source,
                        int(source_volume),
                        frames_per_volume,
                        z_start_frame,
                        z_end_frame,
                        reverse_z_by_volume_parity,
                    )
                )
                if volume.shape != reference_shape:
                    raise ValueError(
                        f"Volume {source_volume} has shape {volume.shape}; "
                        f"expected {reference_shape}"
                    )
                if align_xy:
                    center_xy, rotation_xy = alignment_by_volume[int(source_volume)]
                    volume = align_volume_xy(
                        volume, center_xy, rotation_xy, interpolation_order
                    )
                if flip_x:
                    volume = volume[:, :, ::-1]
                if flip_y:
                    volume = volume[:, ::-1, :]
                volume = volume[:, y0:y1, x0:x1]
                data[local_t, 0] = _to_uint8(
                    volume, intensity_min, calibrated_max
                )

        os.replace(temporary, output_path)
        return shape, calibrated_max
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        source.__exit__(None, None, None)


def _find_point_dataset(group: h5py.Group) -> h5py.Dataset | None:
    for name in POINT_DATASET_CANDIDATES:
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def _point_frames(h5_file: h5py.File) -> list[str]:
    frames = [
        name
        for name, item in h5_file.items()
        if isinstance(item, h5py.Group) and _find_point_dataset(item) is not None
    ]
    return sorted(frames, key=_natural_key)


def load_dynamics_data(
    dynamics_path: Path,
    first_volume: int = 0,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Load points and per-volume XY alignment transforms from dynamics.h5.

    Numeric group names are interpreted as source volume IDs. Other valid
    point groups use their naturally sorted position plus ``first_volume``.
    Groups without a supported point dataset, such as warm-up metadata, are
    ignored.
    """

    frames: list[np.ndarray] = []
    source_volumes: list[int] = []
    centers_xy: list[np.ndarray] = []
    rotations_xy: list[np.ndarray] = []
    with h5py.File(dynamics_path, "r") as dynamics:
        frame_names = _point_frames(dynamics)
        if not frame_names:
            raise ValueError(
                f"No supported neuron point datasets found in {dynamics_path}"
            )

        expected_shape: tuple[int, ...] | None = None
        for frame_index, frame_name in enumerate(frame_names):
            dataset = _find_point_dataset(dynamics[frame_name])
            if dataset is None:
                continue
            points = np.asarray(dataset[:], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 3:
                raise ValueError(
                    f"Point dataset {dataset.name} must have shape (N,F>=3), "
                    f"got {points.shape}"
                )
            if expected_shape is None:
                expected_shape = points.shape
            elif points.shape != expected_shape:
                raise ValueError(
                    f"Point dataset {dataset.name} has shape {points.shape}; "
                    f"expected {expected_shape}"
                )

            group = dynamics[frame_name]
            if CENTER_DATASET not in group or ROTATION_DATASET not in group:
                raise KeyError(
                    f"Dynamics group {group.name} must contain "
                    f"'{CENTER_DATASET}' and '{ROTATION_DATASET}'"
                )
            center = np.asarray(group[CENTER_DATASET][:], dtype=np.float32)
            rotation = np.asarray(group[ROTATION_DATASET][:], dtype=np.float32)
            if center.ndim != 1 or center.size < 2:
                raise ValueError(
                    f"{group.name}/{CENTER_DATASET} must contain at least XY, "
                    f"got shape {center.shape}"
                )
            if rotation.shape != (2, 2):
                raise ValueError(
                    f"{group.name}/{ROTATION_DATASET} must have shape (2,2), "
                    f"got {rotation.shape}"
                )
            if not np.isfinite(center[:2]).all() or not np.isfinite(rotation).all():
                raise ValueError(f"Non-finite alignment transform in {group.name}")
            if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-4):
                raise ValueError(f"Rotation matrix is not orthonormal in {group.name}")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
                raise ValueError(f"Rotation determinant is not +1 in {group.name}")

            frames.append(points)
            centers_xy.append(center[:2])
            rotations_xy.append(rotation)
            source_volumes.append(
                int(frame_name) if frame_name.isdigit() else first_volume + frame_index
            )

    if len(set(source_volumes)) != len(source_volumes):
        raise ValueError("dynamics.h5 contains duplicate source volume IDs")
    return (
        np.stack(frames, axis=0),
        source_volumes,
        np.stack(centers_xy, axis=0),
        np.stack(rotations_xy, axis=0),
    )


def load_neuron_pt_tuple_matrix(
    dynamics_path: Path,
    first_volume: int = 0,
) -> tuple[np.ndarray, list[int]]:
    """Compatibility helper returning only the (T,N,F) point matrix and IDs."""

    points, source_volumes, _, _ = load_dynamics_data(dynamics_path, first_volume)
    return points, source_volumes


def _coordinates_xyz(points: np.ndarray, coordinate_order: str) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Neuron points must have shape (N,F>=3), got {points.shape}")
    order = coordinate_order.lower()
    indices = [order.index(axis) for axis in "xyz"]
    return np.asarray(points[:, indices], dtype=np.float32)


def determine_crop_bounds(
    neuron_pt_tuple: np.ndarray,
    dynamics_volume_numbers: Sequence[int],
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
    selected_volumes: Sequence[int],
    full_image_shape_yx: Sequence[int],
    coordinate_order: str,
    align_xy: bool,
    flip_x: bool,
    flip_y: bool,
    crop_mode: str,
    crop_padding: float,
    custom_crop_xyxy: Sequence[int] | None,
) -> tuple[int, int, int, int]:
    """Return one common post-transform crop as (x0,y0,x1,y1)."""

    height, width = (int(value) for value in full_image_shape_yx)
    mode = crop_mode.strip().lower()
    if mode not in {"none", "point_cloud", "custom"}:
        raise ValueError("crop_mode must be 'none', 'point_cloud', or 'custom'")
    if not np.isfinite(crop_padding) or crop_padding < 0:
        raise ValueError("crop_padding must be a finite non-negative number")

    volume_to_matrix_index = {
        int(volume): index for index, volume in enumerate(dynamics_volume_numbers)
    }
    transformed_frames: list[np.ndarray] = []
    for source_volume in selected_volumes:
        matrix_index = volume_to_matrix_index[int(source_volume)]
        xyz = _coordinates_xyz(neuron_pt_tuple[matrix_index], coordinate_order)
        center_xy, rotation_xy = alignment_by_volume[int(source_volume)]
        transformed = transform_points_xy(
            xyz[:, :2],
            center_xy,
            rotation_xy,
            (height, width),
            align_xy,
            flip_x,
            flip_y,
        )
        finite = np.isfinite(transformed).all(axis=1)
        if finite.any():
            transformed_frames.append(transformed[finite])

    if not transformed_frames:
        raise ValueError("No finite XY point coordinates found for crop calculation")
    all_xy = np.concatenate(transformed_frames, axis=0)

    if mode == "none":
        bounds = (0, 0, width, height)
    elif mode == "custom":
        if custom_crop_xyxy is None or len(custom_crop_xyxy) != 4:
            raise ValueError(
                "custom_crop_xyxy must contain (x0,y0,x1,y1) in custom mode"
            )
        if any(int(value) != value for value in custom_crop_xyxy):
            raise ValueError("custom_crop_xyxy values must be integers")
        bounds = tuple(int(value) for value in custom_crop_xyxy)
    else:
        minimum = np.min(all_xy, axis=0)
        maximum = np.max(all_xy, axis=0)
        bounds = (
            max(0, int(np.floor(minimum[0] - crop_padding))),
            max(0, int(np.floor(minimum[1] - crop_padding))),
            min(width, int(np.ceil(maximum[0] + crop_padding)) + 1),
            min(height, int(np.ceil(maximum[1] + crop_padding)) + 1),
        )

    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"Invalid crop bounds {bounds} for full image shape {(height, width)}"
        )
    outside = (
        (all_xy[:, 0] < x0)
        | (all_xy[:, 0] > x1)
        | (all_xy[:, 1] < y0)
        | (all_xy[:, 1] > y1)
    )
    if outside.any():
        outside_count = int(outside.sum())
        raise ValueError(
            f"Crop bounds {bounds} exclude {outside_count} selected point(s); "
            "increase padding or enlarge the custom crop"
        )
    return bounds


def _worldline_colors(worldline_ids: np.ndarray) -> np.ndarray:
    colors = []
    count = max(len(worldline_ids), 1)
    for index in range(len(worldline_ids)):
        red, green, blue = colorsys.hsv_to_rgb(index / count, 0.75, 1.0)
        colors.append(
            f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
        )
    return np.asarray(colors, dtype="S7")


def create_annotations_h5(
    neuron_pt_tuple: np.ndarray,
    dynamics_volume_numbers: Sequence[int],
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
    annotations_path: Path,
    worldlines_path: Path,
    selected_volumes: Sequence[int],
    full_image_shape_zyx: Sequence[int],
    crop_bounds_xyxy: Sequence[int],
    coordinate_order: str,
    z_ratio: float,
    align_xy: bool = True,
    flip_x: bool = False,
    flip_y: bool = False,
    provenance: bytes = b"ANTT",
    overwrite: bool = False,
) -> int:
    """Write native annotations/worldlines files and return worldline count."""

    if len(provenance) != 4:
        raise ValueError("ZephIR provenance must contain exactly four bytes")
    if neuron_pt_tuple.ndim != 3 or neuron_pt_tuple.shape[2] < 3:
        raise ValueError(
            "neuron_pt_tuple must have shape (T,N,F>=3), got "
            f"{neuron_pt_tuple.shape}"
        )
    if neuron_pt_tuple.shape[0] != len(dynamics_volume_numbers):
        raise ValueError("Dynamics matrix and source volume mapping lengths differ")

    depth, full_height, full_width = (
        int(value) for value in full_image_shape_zyx
    )
    x0, y0, x1, y1 = (int(value) for value in crop_bounds_xyxy)
    height = y1 - y0
    width = x1 - x0
    if z_ratio <= 0:
        raise ValueError("z_ratio must be positive")

    records: list[tuple[int, float, float, float, int]] = []
    used_worldlines: set[int] = set()
    volume_to_matrix_index = {
        int(volume): index for index, volume in enumerate(dynamics_volume_numbers)
    }
    for local_t, source_volume in enumerate(selected_volumes):
        try:
            matrix_index = volume_to_matrix_index[int(source_volume)]
        except KeyError as error:
            raise KeyError(
                f"Source volume {source_volume} is absent from dynamics.h5"
            ) from error
        xyz = _coordinates_xyz(neuron_pt_tuple[matrix_index], coordinate_order)
        center_xy, rotation_xy = alignment_by_volume[int(source_volume)]
        xyz[:, :2] = transform_points_xy(
            xyz[:, :2],
            center_xy,
            rotation_xy,
            (full_height, full_width),
            align_xy,
            flip_x,
            flip_y,
        )
        xyz[:, 0] -= x0
        xyz[:, 1] -= y0
        for worldline_id, (x_value, y_value, z_value) in enumerate(xyz):
            if not np.isfinite((x_value, y_value, z_value)).all():
                continue
            x_normalized = float(x_value / width)
            y_normalized = float(y_value / height)
            z_normalized = float(z_value / (z_ratio * depth))
            if not (
                0.0 <= x_normalized <= 1.0
                and 0.0 <= y_normalized <= 1.0
                and 0.0 <= z_normalized <= 1.0
            ):
                raise ValueError(
                    f"Out-of-volume coordinate at source volume {source_volume}, "
                    f"worldline {worldline_id}: XYZ=({x_value}, {y_value}, "
                    f"{z_value}), normalized=({x_normalized}, {y_normalized}, "
                    f"{z_normalized})"
                )
            records.append(
                (
                    local_t,
                    x_normalized,
                    y_normalized,
                    z_normalized,
                    worldline_id,
                )
            )
            used_worldlines.add(worldline_id)

    if not records:
        raise ValueError("No finite neuron coordinates were found for selected volumes")

    annotations_temporary = _prepare_output(annotations_path, overwrite)
    worldlines_temporary = _prepare_output(worldlines_path, overwrite)
    try:
        count = len(records)
        with h5py.File(annotations_temporary, "w") as annotations:
            annotations.create_dataset(
                "id", data=np.arange(1, count + 1, dtype=np.uint32)
            )
            annotations.create_dataset(
                "t_idx",
                data=np.asarray([record[0] for record in records], dtype=np.uint32),
            )
            annotations.create_dataset(
                "x", data=np.asarray([record[1] for record in records], dtype=np.float32)
            )
            annotations.create_dataset(
                "y", data=np.asarray([record[2] for record in records], dtype=np.float32)
            )
            annotations.create_dataset(
                "z", data=np.asarray([record[3] for record in records], dtype=np.float32)
            )
            annotations.create_dataset(
                "worldline_id",
                data=np.asarray([record[4] for record in records], dtype=np.uint32),
            )
            annotations.create_dataset(
                "parent_id", data=np.zeros(count, dtype=np.uint32)
            )
            annotations.create_dataset(
                "provenance", data=np.full(count, provenance, dtype="S4")
            )

        worldline_ids = np.asarray(sorted(used_worldlines), dtype=np.uint32)
        with h5py.File(worldlines_temporary, "w") as worldlines:
            worldlines.create_dataset("id", data=worldline_ids)
            worldlines.create_dataset(
                "name",
                data=np.asarray([str(value) for value in worldline_ids], dtype="S8"),
            )
            worldlines.create_dataset("color", data=_worldline_colors(worldline_ids))

        os.replace(annotations_temporary, annotations_path)
        os.replace(worldlines_temporary, worldlines_path)
        return len(worldline_ids)
    finally:
        if annotations_temporary.exists():
            annotations_temporary.unlink()
        if worldlines_temporary.exists():
            worldlines_temporary.unlink()


def create_metadata_json(
    output_path: Path,
    data_shape: Sequence[int],
    full_image_shape_zyx: Sequence[int],
    crop_bounds_xyxy: Sequence[int],
    crop_mode: str,
    crop_padding: float,
    custom_crop_xyxy: Sequence[int] | None,
    selected_volumes: Sequence[int],
    intensity_min: float,
    intensity_max: float,
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
    pca_rotation_by_volume: dict[int, np.ndarray],
    align_xy: bool,
    flip_x: bool,
    flip_y: bool,
    goal_angle_degrees: float,
    interpolation_order: int,
    overwrite: bool = False,
) -> None:
    temporary = _prepare_output(output_path, overwrite)
    shape_t, shape_c, shape_z, shape_y, shape_x = (
        int(value) for value in data_shape
    )
    full_shape_z, full_shape_y, full_shape_x = (
        int(value) for value in full_image_shape_zyx
    )
    x0, y0, x1, y1 = (int(value) for value in crop_bounds_xyxy)
    metadata = {
        "shape_t": shape_t,
        "shape_c": shape_c,
        "shape_z": shape_z,
        "shape_y": shape_y,
        "shape_x": shape_x,
        "dtype": "uint8",
        "source_volume_numbers": [int(value) for value in selected_volumes],
        "intensity_min": float(intensity_min),
        "intensity_max": float(intensity_max),
        "crop": {
            "mode": crop_mode.strip().lower(),
            "padding_pixels": float(crop_padding),
            "custom_crop_xyxy": (
                None
                if custom_crop_xyxy is None
                else [int(value) for value in custom_crop_xyxy]
            ),
            "bounds_xyxy": [x0, y0, x1, y1],
            "bounds_convention": "x0/y0 inclusive, x1/y1 exclusive",
            "full_shape_zyx": [full_shape_z, full_shape_y, full_shape_x],
            "output_shape_yx": [shape_y, shape_x],
            "point_formula": "cropped_xy = display_xy - [x0, y0]",
            "common_across_selected_volumes": True,
        },
        "xy_alignment": {
            "enabled": bool(align_xy),
            "point_formula": (
                "aligned_xy = (raw_xy - center_xy) @ pca_rot @ "
                "goal_rot + output_center_xy"
            ),
            "pca_positive_y_display_direction": "down",
            "goal_display_direction_before_flip": "left (-X)",
            "final_display_direction": (
                "right (+X)" if flip_x else "left (-X)"
            ),
            "goal_angle_degrees": float(goal_angle_degrees),
            "goal_rot": image_xy_rotation_matrix(goal_angle_degrees).tolist(),
            "output_center_xy": [
                full_shape_x / 2.0,
                full_shape_y / 2.0,
            ],
            "image_interpolation_order": int(interpolation_order),
            "z_unchanged": True,
            "horizontal_flip": {
                "enabled": bool(flip_x),
                "applied_after_alignment": True,
                "point_formula": (
                    "mirrored_x = (full_shape_x - 1) - aligned_x"
                ),
            },
            "vertical_flip": {
                "enabled": bool(flip_y),
                "applied_after_alignment": True,
                "point_formula": (
                    "mirrored_y = (full_shape_y - 1) - aligned_y"
                ),
            },
            "transforms": [
                {
                    "source_volume": int(volume),
                    "center_xy": alignment_by_volume[int(volume)][0].tolist(),
                    "pca_rot": pca_rotation_by_volume[int(volume)].tolist(),
                    "effective_rot": alignment_by_volume[int(volume)][1].tolist(),
                }
                for volume in selected_volumes
            ],
        },
    }
    try:
        temporary.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert_selected_volumes(
    tiff_path: Path,
    dynamics_path: Path,
    output_dir: Path,
    selected_volumes: Sequence[int],
    *,
    frames_per_volume: int,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z_by_volume_parity: Sequence[bool] = (False, False),
    intensity_min: float = 120.0,
    intensity_percentile: float = 99.0,
    intensity_max: float | None = None,
    coordinate_order: str = "xyz",
    xy_pixel_size: float = 0.3,
    z_step_size: float = 1.5,
    dynamics_first_volume: int = 0,
    align_xy: bool = True,
    goal_angle_degrees: float = 90.0,
    flip_x: bool = False,
    flip_y: bool = False,
    crop_mode: str = "none",
    crop_padding: float = 50.0,
    custom_crop_xyxy: Sequence[int] | None = None,
    image_interpolation_order: int = 1,
    compression: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create all files required to open selected volumes in native ZephIR."""

    tiff_path = Path(tiff_path)
    selected = [int(value) for value in selected_volumes]
    _validate_settings(
        tiff_path,
        selected,
        frames_per_volume,
        z_start_frame,
        z_end_frame,
        reverse_z_by_volume_parity,
        coordinate_order,
        image_interpolation_order,
    )
    if xy_pixel_size <= 0 or z_step_size <= 0:
        raise ValueError("Voxel sizes must be positive")
    if not np.isfinite(goal_angle_degrees):
        raise ValueError("goal_angle_degrees must be finite")
    if not Path(dynamics_path).is_file():
        raise FileNotFoundError(f"dynamics.h5 does not exist: {dynamics_path}")
    (
        neuron_pt_tuple,
        dynamics_volume_numbers,
        centers_xy,
        rotations_xy,
    ) = load_dynamics_data(
        Path(dynamics_path), dynamics_first_volume
    )
    goal_rotation = image_xy_rotation_matrix(goal_angle_degrees)
    pca_rotation_by_volume = {
        int(volume): rotations_xy[index]
        for index, volume in enumerate(dynamics_volume_numbers)
    }
    alignment_by_volume = {
        int(volume): (
            centers_xy[index],
            rotations_xy[index] @ goal_rotation,
        )
        for index, volume in enumerate(dynamics_volume_numbers)
    }
    missing_transforms = [
        volume for volume in selected if volume not in alignment_by_volume
    ]
    if missing_transforms:
        raise KeyError(
            f"Selected volumes are absent from dynamics.h5: {missing_transforms}"
        )

    full_image_shape_zyx = _read_volume_shape(
        tiff_path,
        selected[0],
        frames_per_volume,
        z_start_frame,
        z_end_frame,
    )
    crop_bounds_xyxy = determine_crop_bounds(
        neuron_pt_tuple,
        dynamics_volume_numbers,
        alignment_by_volume,
        selected,
        full_image_shape_zyx[1:],
        coordinate_order,
        align_xy,
        flip_x,
        flip_y,
        crop_mode,
        crop_padding,
        custom_crop_xyxy,
    )

    output_dir = Path(output_dir)
    output_names = ("data.h5", "annotations.h5", "worldlines.h5", "metadata.json")
    existing_outputs = [
        output_dir / name
        for name in output_names
        if (output_dir / name).exists()
    ]
    if existing_outputs and not overwrite:
        formatted = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Output files already exist: {formatted}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}_staging_", dir=output_dir.parent
    ) as staging_name:
        staging_dir = Path(staging_name)
        data_shape, calibrated_intensity_max = create_data_h5(
            tiff_path,
            staging_dir / "data.h5",
            selected,
            frames_per_volume,
            z_start_frame,
            z_end_frame,
            reverse_z_by_volume_parity,
            alignment_by_volume,
            align_xy,
            flip_x,
            flip_y,
            crop_bounds_xyxy,
            image_interpolation_order,
            intensity_min,
            intensity_percentile,
            intensity_max,
            compression,
            False,
        )
        worldline_count = create_annotations_h5(
            neuron_pt_tuple,
            dynamics_volume_numbers,
            alignment_by_volume,
            staging_dir / "annotations.h5",
            staging_dir / "worldlines.h5",
            selected,
            full_image_shape_zyx,
            crop_bounds_xyxy,
            coordinate_order,
            z_step_size / xy_pixel_size,
            align_xy=align_xy,
            flip_x=flip_x,
            flip_y=flip_y,
            provenance=PROVENANCE,
            overwrite=False,
        )
        create_metadata_json(
            staging_dir / "metadata.json",
            data_shape,
            full_image_shape_zyx,
            crop_bounds_xyxy,
            crop_mode,
            crop_padding,
            custom_crop_xyxy,
            selected,
            intensity_min,
            calibrated_intensity_max,
            alignment_by_volume,
            pca_rotation_by_volume,
            align_xy,
            flip_x,
            flip_y,
            goal_angle_degrees,
            image_interpolation_order,
            overwrite=False,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in output_names:
            os.replace(staging_dir / name, output_dir / name)

    result = {
        "output_dir": str(output_dir),
        "data_shape": data_shape,
        "source_volume_numbers": selected,
        "worldline_count": worldline_count,
        "intensity_max": calibrated_intensity_max,
        "xy_aligned": align_xy,
        "goal_angle_degrees": float(goal_angle_degrees),
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
        "crop_mode": crop_mode.strip().lower(),
        "crop_bounds_xyxy": crop_bounds_xyxy,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    convert_selected_volumes(
        TIFF_PATH,
        DYNAMICS_PATH,
        OUTPUT_DIR,
        SELECTED_VOLUMES,
        frames_per_volume=FRAMES_PER_VOLUME,
        z_start_frame=Z_START_FRAME,
        z_end_frame=Z_END_FRAME,
        reverse_z_by_volume_parity=REVERSE_Z_BY_VOLUME_PARITY,
        intensity_min=INTENSITY_MIN,
        intensity_percentile=INTENSITY_PERCENTILE,
        intensity_max=INTENSITY_MAX,
        coordinate_order=COORDINATE_ORDER,
        xy_pixel_size=XY_PIXEL_SIZE,
        z_step_size=Z_STEP_SIZE,
        dynamics_first_volume=DYNAMICS_FIRST_VOLUME,
        align_xy=ALIGN_XY,
        goal_angle_degrees=GOAL_ANGLE_DEGREES,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        crop_mode=CROP_MODE,
        crop_padding=CROP_PADDING,
        custom_crop_xyxy=CUSTOM_CROP_XYXY,
        image_interpolation_order=IMAGE_INTERPOLATION_ORDER,
        compression=H5_COMPRESSION,
        overwrite=OVERWRITE,
    )


if __name__ == "__main__":
    main()
