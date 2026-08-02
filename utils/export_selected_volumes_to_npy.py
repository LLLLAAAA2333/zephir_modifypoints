"""Export SELECTED_VOLUMES as a (T, Z, Y, X) float32 .npy file.

Reads each source volume from TIFF, applies the same XY-alignment, flips, and
crop used by ``create_native_zephir_from_tiff_and_dynamics.py``, then stacks
them into a single float32 array and saves it as a .npy file.

Usage
-----
1. Edit the import below to point to your config source, or set the variables
   directly in ``main()``.
2. Run::

    python utils/export_selected_volumes_to_npy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

# --- helper functions from the main creation script --------------------------
from utils.create_native_zephir_from_tiff_and_dynamics import (  # noqa: E402
    TiffFrameSource,
    _coordinates_xyz,
    _read_volume,
    _read_volume_shape,
    _to_uint8,
    align_volume_xy,
    determine_crop_bounds,
    image_xy_rotation_matrix,
    load_dynamics_data,
    transform_points_xy,
)

# Z coordinate scale factor: Z in dynamics.h5 is stored in XY-pixel units
# (slice_index * Z_STEP_SIZE / XY_PIXEL_SIZE). To recover the slice index,
# divide by this ratio.
XY_PIXEL_SIZE = 0.3
Z_STEP_SIZE = 1.5
Z_SCALE_RATIO = Z_STEP_SIZE / XY_PIXEL_SIZE  # 5.0

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# Reference/red TIFF source.
# TIFF_PATH = Path(r"\\192.168.1.192\Ikrma-2\20260304\W3_2026-03-05_01-03-23\0_Camera-Red_VSC-10629")
TIFF_PATH = Path(r"\\192.168.1.192\Ikrma-2\20260304\W3IMMOB_2026-03-05_01-27-19\0_Camera-Red_VSC-10629")
# TIFF_PATH = Path(r"\\192.168.1.192\Ikrma-2\20260417_WEN1277_OptoAct-ASH_closed-loop\w2_2026-04-17_10-48-01\0_Camera-Red_VSC-10629")

# DYNAMICS_PATH = Path(r"Z:\data5\CBMI_inferred_results\Ikrma\proxy\20260304_w3\dynamics.h5")
DYNAMICS_PATH = Path(r"Z:\data5\CBMI_inferred_results\Ikrma\proxy\20260304_w3_immobile\dynamics.h5")
# DYNAMICS_PATH = Path(r"Z:\data5\CBMI_inferred_results\Ikrma\proxy\20260417_w2\dynamics.h5")

# Source volume numbers, in the desired output order.
SELECTED_VOLUMES = [346, 361, 396]

# TIFF acquisition layout.
FRAMES_PER_VOLUME = 20
Z_START_FRAME = 0
Z_END_FRAME = 17
REVERSE_Z_BY_VOLUME_PARITY = (False, False)

# dynamics.h5 reading.
DYNAMICS_FIRST_VOLUME = 0

# XY alignment and flip (applied before crop).
ALIGN_XY = True
GOAL_ANGLE_DEGREES = -90.0
FLIP_X = False
FLIP_Y = False
IMAGE_INTERPOLATION_ORDER = 1

# XY crop.
CROP_MODE = "none"
CROP_PADDING = 50.0
CUSTOM_CROP_XYXY = None

# Point coordinate order in dynamics.h5.
COORDINATE_ORDER = "xyz"

# Intensity rescaling for optional uint8 output.
INTENSITY_MIN = 105
INTENSITY_PERCENTILE = 99.0
INTENSITY_MAX = 400

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_FOlder= r"H:\Process_temporary\WJH\zephir_modifypoints\data\20260304_w3_immobile_npy"
OUTPUT_NPY = rf"{OUTPUT_FOlder}\volumes.npy"
SAVE_UINT8 = False
OUTPUT_NPY_UINT8 = r"H:\Process_temporary\WJH\zephir_modifypoints\data\selected_volumes_uint8.npy"
SKIP_CROP = True
SAVE_MASK = True
OUTPUT_MASK_NPY = rf"{OUTPUT_FOlder}\neuron_mask.npy"
SAVE_POINTS = True
OUTPUT_POINTS_NPY = rf"{OUTPUT_FOlder}\neuron_point_tuple.npy"


def build_alignment_map() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return ``{source_volume: (center_xy, effective_rotation_xy), ...}``."""
    _, dynamics_volume_numbers, centers_xy, rotations_xy = load_dynamics_data(
        DYNAMICS_PATH, DYNAMICS_FIRST_VOLUME,
    )
    goal_rotation = image_xy_rotation_matrix(GOAL_ANGLE_DEGREES)
    return {
        int(volume): (centers_xy[index], rotations_xy[index] @ goal_rotation)
        for index, volume in enumerate(dynamics_volume_numbers)
    }


def main() -> None:
    selected = [int(v) for v in SELECTED_VOLUMES]
    print(f"Source volumes: {selected}")
    print(f"TIFF path:      {TIFF_PATH}")
    print(f"Dynamics path:  {DYNAMICS_PATH}")

    # --- alignment transforms -------------------------------------------------
    alignment_by_volume = build_alignment_map()
    missing = [v for v in selected if v not in alignment_by_volume]
    if missing:
        raise KeyError(f"Volumes missing from dynamics.h5: {missing}")

    # --- determine crop bounds (same logic as the main pipeline) ---------------
    full_shape_zyx = _read_volume_shape(
        TIFF_PATH, selected[0], FRAMES_PER_VOLUME,
        Z_START_FRAME, Z_END_FRAME,
    )
    full_height, full_width = full_shape_zyx[1], full_shape_zyx[2]

    if SKIP_CROP:
        crop_bounds_xyxy = (0, 0, full_width, full_height)
    else:
        neuron_pt_tuple, dynamics_volume_numbers, _, _ = load_dynamics_data(
            DYNAMICS_PATH, DYNAMICS_FIRST_VOLUME,
        )
        crop_bounds_xyxy = determine_crop_bounds(
            neuron_pt_tuple,
            dynamics_volume_numbers,
            alignment_by_volume,
            selected,
            full_shape_zyx[1:],
            COORDINATE_ORDER,
            ALIGN_XY,
            FLIP_X,
            FLIP_Y,
            CROP_MODE,
            CROP_PADDING,
            CUSTOM_CROP_XYXY,
        )
        print(f"Crop bounds:    {crop_bounds_xyxy}")

    x0, y0, x1, y1 = (int(v) for v in crop_bounds_xyxy)

    # --- calibrate intensity if needed for uint8 output -----------------------
    intensity_max = INTENSITY_MAX

    # --- read, transform, collect ---------------------------------------------
    volumes_f32: list[np.ndarray] = []
    volumes_u8: list[np.ndarray] = []
    reference_shape: tuple[int, int, int] | None = None

    with TiffFrameSource(Path(TIFF_PATH)) as source:
        source.__enter__()
        try:
            for local_t, source_vol in enumerate(selected):
                volume = _read_volume(
                    source,
                    source_vol,
                    FRAMES_PER_VOLUME,
                    Z_START_FRAME,
                    Z_END_FRAME,
                    REVERSE_Z_BY_VOLUME_PARITY,
                )
                if reference_shape is None:
                    reference_shape = volume.shape
                elif volume.shape != reference_shape:
                    raise ValueError(
                        f"Volume {source_vol} shape {volume.shape} ≠ "
                        f"{reference_shape}"
                    )

                # --- geometric transforms (same order as create_data_h5) ------
                if ALIGN_XY:
                    center_xy, rotation_xy = alignment_by_volume[source_vol]
                    volume = align_volume_xy(
                        volume, center_xy, rotation_xy, IMAGE_INTERPOLATION_ORDER,
                    )
                if FLIP_X:
                    volume = volume[:, :, ::-1]
                if FLIP_Y:
                    volume = volume[:, ::-1, :]

                # --- crop -----------------------------------------------------
                if not SKIP_CROP:
                    volume = volume[:, y0:y1, x0:x1]

                volumes_f32.append(volume.astype(np.float32))

                if SAVE_UINT8:
                    if intensity_max is None:
                        intensity_max = float(
                            np.percentile(
                                volume[volume > INTENSITY_MIN], INTENSITY_PERCENTILE
                            )
                        )
                    volumes_u8.append(
                        _to_uint8(volume, INTENSITY_MIN, intensity_max)
                    )

                print(
                    f"  [{local_t}] source_volume={source_vol}  "
                    f"shape={volume.shape}  "
                    f"dtype={volume.dtype}  "
                    f"min={volume.min():.1f}  max={volume.max():.1f}"
                )
        finally:
            source.__exit__(None, None, None)

    # --- stack and save -------------------------------------------------------
    stack_f32 = np.stack(volumes_f32, axis=0)  # (T, Z, Y, X)
    print(f"\nFinal float32 stack shape: {stack_f32.shape}  dtype: {stack_f32.dtype}")
    output_path = Path(OUTPUT_NPY)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, stack_f32)
    print(f"Saved: {output_path}")

    if SAVE_UINT8 and volumes_u8:
        stack_u8 = np.stack(volumes_u8, axis=0)
        print(f"Final uint8 stack shape:   {stack_u8.shape}  dtype: {stack_u8.dtype}")
        uint8_path = Path(OUTPUT_NPY_UINT8)
        uint8_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(uint8_path, stack_u8)
        print(f"Saved: {uint8_path}")

    if SAVE_MASK or SAVE_POINTS:
        # Load and transform neuron_pt_tuple once; optionally draw masks and/or
        # save the transformed point cloud.
        neuron_pt_tuple, dynamics_volume_numbers, _, _ = load_dynamics_data(
            DYNAMICS_PATH, DYNAMICS_FIRST_VOLUME,
        )
        volume_to_index = {
            int(v): i for i, v in enumerate(dynamics_volume_numbers)
        }
        z_dim, y_dim, x_dim = reference_shape  # reference_shape = first volume shape
        masks: list[np.ndarray] = []
        transformed_frames: list[np.ndarray] = []

        for local_t, source_vol in enumerate(selected):
            matrix_index = volume_to_index[int(source_vol)]
            points = np.asarray(neuron_pt_tuple[matrix_index], dtype=np.float32).copy()
            if points.size == 0:
                if SAVE_MASK:
                    masks.append(np.zeros((z_dim, y_dim, x_dim), dtype=np.int16))
                if SAVE_POINTS:
                    transformed_frames.append(points)
                continue

            xyz = _coordinates_xyz(points, COORDINATE_ORDER)
            center_xy, rotation_xy = alignment_by_volume[source_vol]
            xyz[:, :2] = transform_points_xy(
                xyz[:, :2],
                center_xy,
                rotation_xy,
                (full_height, full_width),
                ALIGN_XY,
                FLIP_X,
                FLIP_Y,
            )
            xyz[:, 0] -= x0
            xyz[:, 1] -= y0
            points[:, :3] = xyz

            if SAVE_POINTS:
                transformed_frames.append(points)

            if SAVE_MASK:
                mask = np.zeros((z_dim, y_dim, x_dim), dtype=np.int16)
                for neuron_id, (x_val, y_val, z_val) in enumerate(xyz):
                    if np.isnan((x_val, y_val, z_val)).any():
                        continue
                    w = float(points[neuron_id, 3])
                    h = float(points[neuron_id, 4])
                    d_scaled = float(points[neuron_id, 5])
                    if np.isnan((w, h, d_scaled)).any():
                        continue
                    z_centre = int(round(z_val / Z_SCALE_RATIO))
                    depth_layers = max(1, int(np.ceil(d_scaled / Z_SCALE_RATIO)))
                    half_depth = depth_layers / 2.0
                    z_min = max(0, int(np.ceil(z_centre - half_depth)))
                    z_max = min(z_dim, int(np.ceil(z_centre + half_depth)))
                    x_min = max(0, int(x_val - w / 2.0))
                    x_max = min(x_dim, int(x_val + w / 2.0))
                    y_min = max(0, int(y_val - h / 2.0))
                    y_max = min(y_dim, int(y_val + h / 2.0))
                    if z_min < z_max and y_min < y_max and x_min < x_max:
                        mask[z_min:z_max, y_min:y_max, x_min:x_max] = neuron_id + 1
                masks.append(mask)

        if SAVE_MASK:
            stack_mask = np.stack(masks, axis=0)  # (T, Z, Y, X)
            print(f"\nFinal mask stack shape: {stack_mask.shape}  dtype: {stack_mask.dtype}")
            mask_path = Path(OUTPUT_MASK_NPY)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_path, stack_mask)
            print(f"Saved: {mask_path}")

        if SAVE_POINTS:
            stack_points = np.stack(transformed_frames, axis=0)  # (T, N, 8)
            print(f"\nFinal points stack shape: {stack_points.shape}  dtype: {stack_points.dtype}")
            points_path = Path(OUTPUT_POINTS_NPY)
            points_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(points_path, stack_points)
            print(f"Saved: {points_path}")

    print("Done.")


if __name__ == "__main__":
    main()
