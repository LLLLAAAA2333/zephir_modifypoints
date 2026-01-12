# %%
import os
import sys
import json
import re
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.logged_operation import logged_operation

def extract_number_from_filename(filename):
    """从文件名中提取数字，用于排序 (已保留以兼容旧逻辑)."""
    match = re.findall(r'\d+', filename)
    return int(match[-1]) if match else -1


def natural_sort_key(value: str):
    """使用自然序对字符串进行排序，确保多段数字按数值排序."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]

def rescale_image(image, target_min, target_max, source_min = None, source_max = None):
    """
    Rescale the values in an image to a new specified range.This function includes
    checks to ensure that the target and source ranges are valid.
    """
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")
    if source_min is None:
        source_min = np.min(image)
    if source_max is None:
        source_max = np.max(image)
    
    image_float32 = image.astype(np.float32)
    if source_min >= source_max:
        # If all values are the same, return an image with all values at target_min
        if source_min == source_max:
            return np.full(image.shape, target_min, dtype=np.uint8)
        raise ValueError("source_min must be less than source_max")

    return np.clip((image_float32 - source_min) / (source_max - source_min) * (target_max - target_min) + target_min,
                             target_min, target_max).astype(np.uint8)

def create_zephir_data_from_npy(image_folder, zephir_folder, **kwargs):
    """
    从包含.npy文件的文件夹中读取volume数据，转换为ZephIR格式的data.h5
    
    Parameters:
    image_folder: 包含.npy文件的文件夹路径
    zephir_folder: ZephIR输出文件夹路径
    kwargs: 其他参数，包括 denoise_range
    
    Returns:
    int: 处理的时间帧数量
    """
    denoise_range = kwargs.get('denoise_range')
    
    npy_paths = sorted(
        Path(image_folder).glob('*.npy'),
        key=lambda path: natural_sort_key(path.name)
    )
    
    if not npy_paths:
        raise ValueError(f"No .npy files found in {image_folder}")
    
    print(f"Found {len(npy_paths)} .npy files")
    
    all_volumes = []
    
    for npy_path in tqdm(npy_paths, desc="Processing npy files"):
        volume = np.load(npy_path)  # (y, x, z)
        
        if volume is not None and volume.size > 0:
            # (y, x, z) -> (z, y, x)
            volume = np.transpose(volume, (2, 0, 1))
            
            if denoise_range:
                volume[(volume < denoise_range[0]) | (volume > denoise_range[1])] = 0
            
            volume_scaled = rescale_image(volume, 0, 255).astype(np.uint8)
            all_volumes.append(volume_scaled)
        else:
            print(f"Warning: No valid data in {npy_path.name}")
            
    if not all_volumes:
        raise ValueError("No valid volume data found in any .npy files")
        
    vol_num = len(all_volumes)
    slice_num, height, width = all_volumes[0].shape
    
    os.makedirs(zephir_folder, exist_ok=True)
    
    img_data = np.empty((vol_num, 1, slice_num, height, width), dtype=np.uint8)
    for i in range(vol_num):
        img_data[i, 0] = all_volumes[i]
        
    data_path = os.path.join(zephir_folder, 'data.h5')
    with h5py.File(data_path, 'w') as hf:
        hf.create_dataset('data', data=img_data, chunks=True)
        
    print(f"Successfully created data.h5 with {vol_num} timepoints")
    return vol_num, {'width': width, 'height': height, 'depth': slice_num}


def create_zephir_annotations_from_npy(neuron_pt_tuple, zephir_folder, **kwargs):
    """
    input: all neuron_pt_tuple (time x neuron_pt_tuple)
    output: annotations.h5
    """
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    z_ratio = kwargs.get('z_ratio', 5)

    total_t = len(neuron_pt_tuple)
    
    os.makedirs(zephir_folder, exist_ok=True)
    h5file_path = os.path.join(zephir_folder, 'annotations.h5')
    
    if os.path.exists(h5file_path):
        os.remove(h5file_path)
        print('Deleting existing annotations.h5')
        
    try:
        with h5py.File(h5file_path, 'a') as f:
            max_shape = (None,)
            f.create_dataset('/x', shape=(0,), maxshape=max_shape, dtype='float32')
            f.create_dataset('/y', shape=(0,), maxshape=max_shape, dtype='float32')
            f.create_dataset('/z', shape=(0,), maxshape=max_shape, dtype='float32')
            f.create_dataset('/id', shape=(0,), maxshape=max_shape, dtype='uint32')
            f.create_dataset('/parent_id', shape=(0,), maxshape=max_shape, dtype='uint16')
            f.create_dataset('/worldline_id', shape=(0,), maxshape=max_shape, dtype='uint8')
            f.create_dataset('/provenance', shape=(0,), maxshape=max_shape, dtype='S4')
            f.create_dataset('/t_idx', shape=(0,), maxshape=max_shape, dtype='uint16')

            def append_frame(frame_data, t_idx):
                if frame_data is None or frame_data.size == 0:
                    return
                n = frame_data.shape[0]
                if n == 0:
                    return
                new_size = f['/x'].shape[0] + n

                f['/x'].resize(new_size, axis=0)
                f['/x'][-n:] = frame_data[:, 0] / width
                f['/y'].resize(new_size, axis=0)
                f['/y'][-n:] = frame_data[:, 1] / height
                f['/z'].resize(new_size, axis=0)
                f['/z'][-n:] = frame_data[:, 2] / (z_ratio * depth)

                f['/worldline_id'].resize(new_size, axis=0)
                f['/worldline_id'][-n:] = np.arange(0, n)
                f['/provenance'].resize(new_size, axis=0)
                f['/provenance'][-n:] = np.array(['ANTT'] * n, dtype='S4')

                f['/t_idx'].resize(new_size, axis=0)
                f['/t_idx'][-n:] = np.full(n, t_idx, dtype='uint16')

                current_start = new_size - n
                id_values = np.arange(current_start, current_start + n, dtype='uint32')

                f['/id'].resize(new_size, axis=0)
                f['/id'][-n:] = id_values + 1
                f['/parent_id'].resize(new_size, axis=0)
                f['/parent_id'][-n:] = np.full(n, t_idx + 1, dtype='uint16')

            for t_idx in range(total_t):
                neuron_data = neuron_pt_tuple[t_idx]
                append_frame(neuron_data, t_idx)

            print(f'Finished creating annotations.h5 with {total_t} timepoints')

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return None

def create_metadata_json_from_npy(zephir_folder, volume_num, **kwargs):
    """
    为文件夹创建metadata.json文件
    """
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    
    os.makedirs(zephir_folder, exist_ok=True)
    
    metadata = {
        "shape_t": volume_num,
        "shape_c": 1,
        "shape_z": depth,
        "shape_y": height,
        "shape_x": width,
        "dtype": "uint8",
    }
    
    metadata_path = os.path.join(zephir_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f'Created metadata.json for {volume_num} timepoints')

def load_neuron_pt_tuple_from_annotations(annotations_path, return_dict=False, **kwargs):
    """
    Load ZephIR annotations back into a ``neuron_pt_tuple`` tensor or a dictionary.
    
    Args:
        return_dict (bool): If True, return a dictionary {worldline_id: {t_idx: [x, y, z]}}
                            instead of a numpy array.
    """

    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    z_ratio = kwargs.get('z_ratio', 5)

    try:
        with h5py.File(annotations_path, 'r') as f:
            if '/t_idx' not in f:
                print(f"Error: '/t_idx' not found in {annotations_path}")
                return None

            t_indices = f['/t_idx'][:]
            if t_indices.size == 0:
                 if return_dict:
                     return {}
                 return np.empty((0, 0, 8), dtype=np.float32)

            worldline_ids = f['/worldline_id'][:]
            x = f['/x'][:] * width
            y = f['/y'][:] * height
            z = f['/z'][:] * (z_ratio * depth)
            
            # --- Dictionary Mode ---
            if return_dict:
                valid_ids = np.unique(worldline_ids)
                modify_neuron_coords_dict = {}
                
                # Pre-initialize dictionaries for each valid ID
                for wl_id in valid_ids:
                    modify_neuron_coords_dict[wl_id] = {}
                
                for i in range(len(t_indices)):
                    t = t_indices[i]
                    # Note: ZephIR usually uses uint32 for IDs, convert to int for standard dict keys
                    neuron_idx = int(worldline_ids[i]) 
                    
                    modify_neuron_coords_dict[neuron_idx][t] = [x[i], y[i], z[i]]
                    
                print(f"Loaded modifications for {len(modify_neuron_coords_dict)} tracks from {annotations_path}")
                return modify_neuron_coords_dict

            # --- Array Mode (Original Logic) ---
            max_t = t_indices.max()
            max_neurons = worldline_ids.max() + 1
            
            neuron_pt_tuple = np.full((max_t + 1, max_neurons, 8), np.nan, dtype=np.float32)

            for i in range(len(t_indices)):
                t = t_indices[i]
                neuron_idx = worldline_ids[i]
                neuron_pt_tuple[t, neuron_idx, 0] = x[i]
                neuron_pt_tuple[t, neuron_idx, 1] = y[i]
                neuron_pt_tuple[t, neuron_idx, 2] = z[i]
        
        print(f"Successfully loaded {neuron_pt_tuple.shape[0]} frames and {neuron_pt_tuple.shape[1]} neurons from {annotations_path}")
        return neuron_pt_tuple

    except Exception as e:
        print(f"Error loading annotations from {annotations_path}: {e}")
        return None

def convert_annotations_to_neuron_pt_tuple(
    annotations_path,
    output_path,
    template_neuron_pt_tuple_path=None,
    **kwargs,
):
    """
    Convert ZephIR annotations.h5 back to neuron_pt_tuple.npy format. 
    Handles deleted tracks by removing them from the template.
    """
    
    # 1. Load Modifications as Dictionary
    modify_neuron_coords_dict = load_neuron_pt_tuple_from_annotations(
        annotations_path,
        return_dict=True,
        **kwargs,
    )

    if modify_neuron_coords_dict is None:
        raise ValueError("Failed to load neuron_pt_tuple data from annotations.")

    # 2. Load Template
    if template_neuron_pt_tuple_path is None:
        raise ValueError("template_neuron_pt_tuple_path is required for this operation.")
        
    template_neuron_pt_tuple = np.load(template_neuron_pt_tuple_path)
    print(f"Loaded template neuron_pt_tuple from {template_neuron_pt_tuple_path}")

    # Handle Template Dimensions
    if template_neuron_pt_tuple.ndim == 2:
        # Expand to 3D: (1, neurons, features)
        template_neuron_pt_tuple = template_neuron_pt_tuple[np.newaxis, :, :]
    
    template_time, template_neurons, template_features = template_neuron_pt_tuple.shape

    # 3. Filter and Reconstruct
    # Valid IDs are those present in the annotations (modify_neuron_coords_dict)
    valid_ids = sorted(list(modify_neuron_coords_dict.keys()))
    new_neuron_count = len(valid_ids)
    
    print(f"Reconstructing neuron_pt_tuple: {template_neurons} -> {new_neuron_count} neurons")
    
    # Create new array with same time and features, but filtered neuron count
    new_neuron_pt_tuple = np.full(
        (template_time, new_neuron_count, template_features), 
        np.nan, 
        dtype=np.float32
    )

    # 4. Map Data
    for new_idx, original_id in enumerate(valid_ids):
        # A. Copy from Template (if original ID exists in template)
        if original_id < template_neurons:
            new_neuron_pt_tuple[:, new_idx, :] = template_neuron_pt_tuple[:, original_id, :]
        else:
            # New track that didn't exist in template
            tail_dim = max(template_features - 3, 0)
            if tail_dim > 0:
                fallback_tail_values = np.array([14.0, 14.0, 15.0, 0.0, 0.0], dtype=np.float32)
                fallback_tail = np.zeros(tail_dim, dtype=np.float32)
                copy_len = min(tail_dim, fallback_tail_values.size)
                fallback_tail[:copy_len] = fallback_tail_values[:copy_len]
                
                # Assign to all timepoints for this new neuron
                new_neuron_pt_tuple[:, new_idx, 3:] = fallback_tail

        # B. Apply Modifications from Annotations
        track_mods = modify_neuron_coords_dict[original_id]
        for t, coords in track_mods.items():
            if t < template_time:
                # Update XYZ (indices 0, 1, 2)
                new_neuron_pt_tuple[t, new_idx, 0:3] = coords
    
    # 5. Save
    np.save(output_path, new_neuron_pt_tuple.squeeze())
    print(f"Converted and cleaned neuron_pt_tuple saved to {output_path}")


# %%
if __name__ == "__main__":
    # Example usage:
    image_folder = r"Z:\data5\WJH\olfactory\20251111\w6\ref_volumes"
    neuron_pt_tuple_path = r"Z:\data5\WJH\olfactory\20251111\w6\output\reference_inference_results\ref_neuron_pt_tuple_filled.npy"
    zephir_folder = r"H:\Process_temporary\WJH\zephir_modifypoints\data\20251011\w6"
    
    xoy_unit = 0.3
    z_unit = 1.5
    z_ratio = z_unit / xoy_unit
    
    params = dict(
        z_ratio=z_ratio,
        denoise_range=(120, 1000),
    )
    
    convert_npy_to_ZephIR_format(image_folder, neuron_pt_tuple_path, zephir_folder, **params)