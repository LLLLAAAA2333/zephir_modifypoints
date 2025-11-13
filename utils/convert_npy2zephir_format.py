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

def load_neuron_pt_tuple_from_annotations(annotations_path, **kwargs):
    """Load ZephIR annotations back into a ``neuron_pt_tuple`` tensor."""

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
                return np.empty((0, 0, 8), dtype=np.float32)

            max_t = t_indices.max()
            worldline_ids = f['/worldline_id'][:]
            max_neurons = worldline_ids.max() + 1
            
            neuron_pt_tuple = np.full((max_t + 1, max_neurons, 8), np.nan, dtype=np.float32)

            x = f['/x'][:] * width
            y = f['/y'][:] * height
            z = f['/z'][:] * (z_ratio * depth)

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

def convert_npy_to_ZephIR_format(image_folder, neuron_pt_tuple_path, zephir_folder, **params):
    """
    将NPY数据转换为ZephIR格式
    
    Parameters:
    image_folder: 包含.npy图像文件的文件夹
    neuron_pt_tuple_path: neuron_pt_tuple.npy文件路径
    zephir_folder: ZephIR输出文件夹路径
    params: 其他参数
    """
    # 1. Load neuron_pt_tuple
    neuron_pt_tuple = np.load(neuron_pt_tuple_path)
    print(f"Loaded neuron_pt_tuple from {neuron_pt_tuple_path} with shape {neuron_pt_tuple.shape}")
    
    # 2. Create data.h5
    volume_number, shape_params = create_zephir_data_from_npy(image_folder, zephir_folder, **params)
    params.update(shape_params)

    # 3. Create annotations.h5
    create_zephir_annotations_from_npy(neuron_pt_tuple, zephir_folder, **params)
    
    # 4. Create metadata.json
    create_metadata_json_from_npy(zephir_folder, volume_number, **params)

    print("\nConversion to ZephIR format complete.")
    
    # # 5. Convert back and save example
    # annotations_path = os.path.join(zephir_folder, 'annotations.h5')
    # output_npy_path = os.path.join(zephir_folder, 'neuron_pt_tuple_reverted.npy')
    
    # reverted_neuron_pt_tuple = load_neuron_pt_tuple_from_annotations(annotations_path, **params)
    
    # if reverted_neuron_pt_tuple is not None:
    #     np.save(output_npy_path, reverted_neuron_pt_tuple)
    #     print(f"Reverted neuron_pt_tuple saved to {output_npy_path}")
def convert_annotations_to_neuron_pt_tuple(
    annotations_path,
    output_path,
    template_neuron_pt_tuple_path=None,
    **kwargs,
):
    """
    Convert ZephIR annotations.h5 back to neuron_pt_tuple.npy format.

    Parameters:
    annotations_path: 输入的annotations.h5文件路径
    output_path: 输出的neuron_pt_tuple文件路径 (.npy或.h5)
    template_neuron_pt_tuple_path: 模板文件路径(.npy或.h5)，与上者二选一即可
    kwargs: 包含width, height, depth, z_ratio等参数
    """
    neuron_pt_tuple_xyz = load_neuron_pt_tuple_from_annotations(
        annotations_path,
        **kwargs,
    )

    if template_neuron_pt_tuple_path is not None:
        template_neuron_pt_tuple = np.load(template_neuron_pt_tuple_path)
        neuron_pt_tuple_xyz[:,:,3:] = template_neuron_pt_tuple[:,:,3:]
        print(f"Loaded template neuron_pt_tuple from {template_neuron_pt_tuple_path}")
    
    np.save(output_path, neuron_pt_tuple_xyz)
    print(f"Converted neuron_pt_tuple saved to {output_path}")


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