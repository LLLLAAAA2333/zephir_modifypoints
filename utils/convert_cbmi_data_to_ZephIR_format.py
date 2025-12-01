# %%
# import packages
import sys
import h5py
import numpy as np
import os
import re
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.MatToolkit import extract_number_from_filename, get_matlab_file_info, load_single_timepoint_from_matlab
from utils.logged_operation import logged_operation
from tqdm import tqdm
#%%
def load_neuron_pt_tuple(infer_result):
    with h5py.File(infer_result, 'r') as f:
        all_groups = list(f.keys())
        if all_groups[-1] == 'Structure':
            all_groups = all_groups[:-1]
        group_len  = len(all_groups)
        num_neurons = f[all_groups[0]]['d_neuron_pt_tuple_matched_raw_vol'].shape[0]
        neuron_pt_tuple = np.empty((group_len, num_neurons, 8), dtype=np.float32)
        for frame_idx, frame_name in enumerate(all_groups):
            neuron_pt_tuple[frame_idx] = f[frame_name]['d_neuron_pt_tuple_matched_raw_vol'][:]
    
    return neuron_pt_tuple, group_len
   

def rescale_image(image, target_min, target_max, source_min = None, source_max = None):
    """
    Rescale the values in an image to a new specified range.This function includes
    checks to ensure that the target and source ranges are valid.
    Parameters:
    image (numpy.ndarray): The input image array with pixel values.
    target_min : The minimum value of the target range.
    target_max : The maximum value of the target range.
    source_min (optional): The minimum value of the image's original range.
                           If None, it is automatically computed from the image.
    source_max (optional): The maximum value of the image's original range.
                           If None, it is automatically computed from the image.
    Returns:
    numpy.ndarray: The rescaled image array where the original image values have been
                   scaled to fit within the new target range, while ensuring that all
                   values lie within this range using clipping.
    Raises:
    ValueError: If the target or source ranges are invalid (i.e., min is not less than max).
    """
    # Check that target_min is less than target_max
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")
    # If source_min or source_max are not provided, compute them from the image
    if source_min is None:
        source_min = np.min(image)
    if source_max is None:
        original_max = np.max(image)
        if original_max >= 1000:
            source_max = np.max(image[image < 1000])
        else:
            source_max = original_max
    
    # Check that source_min is less than source_max
    if source_min >= source_max:
        raise ValueError("source_min must be less than source_max")

    # Memory efficient implementation using in-place operations
    image_float32 = image.astype(np.float32)
    
    scale = (target_max - target_min) / (source_max - source_min)
    offset = target_min - source_min * scale
    
    # In-place operations to save memory
    np.multiply(image_float32, scale, out=image_float32)
    np.add(image_float32, offset, out=image_float32)
    np.clip(image_float32, target_min, target_max, out=image_float32)
    
    return image_float32.astype(image.dtype)


def create_zephir_data(mat_folder_path, zephir_folder, vol_num, root_t_index=None, **kwargs):
    """
    从包含.mat文件的文件夹中读取cell array数据，转换为ZephIR格式的data.h5
    每100个时间帧保存为一个独立的data.h5文件，存放在独立的子文件夹中
    
    Parameters:
    mat_folder_path: 包含.mat文件的文件夹路径
    zephir_folder: ZephIR输出文件夹路径
    vol_num: 处理的时间帧数量
    root_t_index: 根volume的绝对时间索引，会在每个chunk的首帧插入
    kwargs: 其他参数，包括volume_shape, denoise_range, chunk_size等
    
    Returns:
    int: 处理的时间帧数量
    """
    # 从kwargs中提取参数
    volume_shape = kwargs.get('volume_shape', (1024, 1024, 18))
    denoise_range = kwargs.get('denoise_range', (120, 1000))
    chunk_size = kwargs.get('chunk_size', 100)  # 每个文件包含的时间帧数量
    if root_t_index is None and 'root_t_index' in kwargs:
        root_t_index = kwargs.get('root_t_index')
        
    # 获取所有.mat文件并排序
    mat_files = [file for file in os.listdir(mat_folder_path) if file.endswith('.mat')]
    mat_files.sort(key=extract_number_from_filename)
    
    if not mat_files:
        raise ValueError(f"No .mat files found in {mat_folder_path}")
    
    print(f"Found {len(mat_files)} MATLAB files")
    
    # 预扫描：计算总时间点并定位 root_volume
    total_timepoints = 0
    mat_file_infos = []
    root_volume = None
    
    # 第一次遍历：获取文件信息并找到 root_volume
    current_t = 0
    root_found = False
    
    print("Scanning files...")
    for mat_file in mat_files:
        filepath = os.path.join(mat_folder_path, mat_file)
        file_info = get_matlab_file_info(filepath)
        mat_file_infos.append(file_info)
        
        if file_info:
            if file_info['is_cell_array']:
                n_t = file_info['shape'][1]
            else:
                n_t = 1
            
            # 如果需要 root_volume 且尚未加载，检查是否在当前文件中
            if root_t_index is not None and not root_found:
                if current_t <= root_t_index < current_t + n_t:
                    local_t = root_t_index - current_t
                    print(f"Loading root volume from {mat_file} (t={local_t})...")
                    vol = load_single_timepoint_from_matlab(filepath, local_t)
                    if vol is not None:
                        vol = np.transpose(vol, (0, 2, 1))
                        if denoise_range:
                            vol[(vol < denoise_range[0]) | (vol > denoise_range[1])] = 0
                        root_volume = rescale_image(vol, 0, 255).astype(np.uint8)
                        root_found = True
            
            total_timepoints += n_t
            current_t += n_t
            
    print(f"Total timepoints to process: {total_timepoints}")
    if root_t_index is not None and root_volume is None:
        raise ValueError(f"root_t_index {root_t_index} is out of range or data could not be loaded")

    # 创建主文件夹
    os.makedirs(zephir_folder, exist_ok=True)
    
    # 准备处理变量
    current_chunk_volumes = []
    chunk_idx = 0
    global_processed_count = 0
    
    # 辅助函数：写入一个 chunk
    def write_chunk(volumes, c_idx, start_t):
        end_t = start_t + len(volumes)
        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        chunk_vol_num = len(volumes)
        root_offset = 1 if root_volume is not None else 0
        total_frames = chunk_vol_num + root_offset
        
        slice_num = volumes[0].shape[0]
        height, width = volumes[0].shape[1], volumes[0].shape[2]
        
        img_data = np.empty((total_frames, 1, slice_num, height, width), dtype=np.uint8)
        
        frame_offset = 0
        if root_volume is not None:
            img_data[0, 0] = root_volume
            frame_offset = 1
            
        for i in range(chunk_vol_num):
            img_data[frame_offset + i, 0] = volumes[i]
            
        data_path = os.path.join(subfolder_path, 'data.h5')
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset('data', data=img_data, chunks=True)
            
        if root_volume is not None:
            print(f"Created {subfolder_name}/data.h5 (t={start_t} to {end_t-1}, root_t={root_t_index})")
        else:
            print(f"Created {subfolder_name}/data.h5 (t={start_t} to {end_t-1})")

    # 第二次遍历：逐个处理并分批写入
    for i, mat_file in enumerate(tqdm(mat_files, desc="Processing files")):
        if global_processed_count >= vol_num:
            break
            
        with logged_operation(display_context=f"Processing {mat_file} ({i+1}/{len(mat_files)})"):
            file_info = mat_file_infos[i]
            if file_info is None: continue
            
            filepath = os.path.join(mat_folder_path, mat_file)
            num_timepoints = file_info['shape'][1] if file_info['is_cell_array'] else 1
            
            for timepoint_idx in range(num_timepoints):
                if global_processed_count >= vol_num:
                    break
                    
                volume = load_single_timepoint_from_matlab(filepath, timepoint_idx)
                
                if volume is not None and volume.size > 0:
                    volume = np.transpose(volume, (0, 2, 1))
                    if denoise_range:
                        volume[(volume < denoise_range[0]) | (volume > denoise_range[1])] = 0
                    
                    # 此时内存中只保留当前 chunk 的数据
                    volume_scaled = rescale_image(volume, 0, 255).astype(np.uint8)
                    current_chunk_volumes.append(volume_scaled)
                    global_processed_count += 1
                    
                    # 如果当前 chunk 满了，写入磁盘并清空内存
                    if len(current_chunk_volumes) == chunk_size:
                        start_t = chunk_idx * chunk_size
                        write_chunk(current_chunk_volumes, chunk_idx, start_t)
                        current_chunk_volumes = [] # 释放内存
                        chunk_idx += 1
                else:
                    print(f"Warning: No valid data at timepoint {timepoint_idx} in {mat_file}")

    # 处理最后一个不满的 chunk
    if current_chunk_volumes:
        start_t = chunk_idx * chunk_size
        write_chunk(current_chunk_volumes, chunk_idx, start_t)
        chunk_idx += 1

    if root_volume is not None:
        print(f"Total: Created {chunk_idx} chunks with {global_processed_count} timepoints plus shared root")
    else:
        print(f"Total: Created {chunk_idx} chunks with {global_processed_count} timepoints")
        
    return global_processed_count
    

def create_zephir_annotations(neuron_pt_tuple, zephir_folder_path, root_t_index=None, **kwargs):
    '''
    input: all neuron_pt_tuple (time x neuron_pt_tuple)
    output: annotations.h5 (分chunk保存在各个子文件夹中)
    '''
    # get parameter
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    z_ratio = kwargs.get('z_ratio', 5)
    chunk_size = kwargs.get('chunk_size', 100)  # 每个文件包含的时间帧数量
    if root_t_index is None and 'root_t_index' in kwargs:
        root_t_index = kwargs.get('root_t_index')

    total_t = len(neuron_pt_tuple)
    num_chunks = (total_t + chunk_size - 1) // chunk_size  # 向上取整
    root_data = None
    if root_t_index is not None:
        root_t_index = int(root_t_index)
        if not (0 <= root_t_index < total_t):
            raise ValueError(f"root_t_index {root_t_index} is out of range for {total_t} timepoints")
        root_data = neuron_pt_tuple[root_t_index]
    
    # 按chunk分批处理
    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * chunk_size
        end_t = min(start_t + chunk_size, total_t)
        
        # 创建子文件夹名称
        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # 当前chunk的annotations文件路径
        h5file_path = os.path.join(subfolder_path, 'annotations.h5')
        
        if os.path.exists(h5file_path):
            os.remove(h5file_path)
            print(f'Deleting existing {subfolder_name}/annotations.h5')
        
        try:
            with h5py.File(h5file_path, 'a') as f:
                if '/x' not in f:
                    max_shape = (None,)
                    f.create_dataset('/x', shape=(0,), maxshape=max_shape, dtype='float32')
                    f.create_dataset('/y', shape=(0,), maxshape=max_shape, dtype='float32')
                    f.create_dataset('/z', shape=(0,), maxshape=max_shape, dtype='float32')
                    f.create_dataset('/id', shape=(0,), maxshape=max_shape, dtype='uint32')
                    f.create_dataset('/parent_id', shape=(0,), maxshape=max_shape, dtype='uint16')
                    f.create_dataset('/worldline_id', shape=(0,), maxshape=max_shape, dtype='uint8')
                    f.create_dataset('/provenance', shape=(0,), maxshape=max_shape, dtype='S4')
                    f.create_dataset('/t_idx', shape=(0,), maxshape=max_shape, dtype='uint16')
                    f.create_dataset('/abs_t_idx', shape=(0,), maxshape=max_shape, dtype='uint16')  # 绝对时间索引

                def append_frame(frame_data, local_t_idx, abs_t_idx):
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
                    f['/t_idx'][-n:] = np.full(n, local_t_idx, dtype='uint16')

                    f['/abs_t_idx'].resize(new_size, axis=0)
                    f['/abs_t_idx'][-n:] = np.full(n, abs_t_idx, dtype='uint16')

                    current_start = new_size - n
                    id_values = np.arange(current_start, current_start + n, dtype='uint32')

                    f['/id'].resize(new_size, axis=0)
                    f['/id'][-n:] = id_values + 1
                    f['/parent_id'].resize(new_size, axis=0)
                    f['/parent_id'][-n:] = np.full(n, local_t_idx + 1, dtype='uint16')

                root_offset = 1 if root_data is not None else 0
                if root_data is not None:
                    append_frame(root_data, 0, root_t_index)

                # 处理当前chunk内的每个时间帧
                for local_t_idx in range(end_t - start_t):
                    global_t_idx = start_t + local_t_idx  # 全局时间索引
                    neuron_data = neuron_pt_tuple[global_t_idx]
                    append_frame(neuron_data, local_t_idx + root_offset, global_t_idx)

                if root_data is not None:
                    print(f'Finished creating {subfolder_name}/annotations.h5 with root_t={root_t_index} (t={start_t} to {end_t-1})')
                else:
                    print(f'Finished creating {subfolder_name}/annotations.h5 (t={start_t} to {end_t-1})')

        except Exception as e:
            print(f"An error occurred in {subfolder_name}: {e}")
            return None

    if root_data is not None:
        print(f'Total: Created {num_chunks} annotation chunks with shared root_t={root_t_index}')
    else:
        print(f'Total: Created {num_chunks} annotation chunks')
    return None


import json
def create_metadata_json(zephir_folder_path, volume_num, root_t_index=None, **kwargs):
    """
    为每个chunk子文件夹创建metadata.json文件
    """
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    chunk_size = kwargs.get('chunk_size', 100)
    if root_t_index is None and 'root_t_index' in kwargs:
        root_t_index = kwargs.get('root_t_index')
    root_offset = 1 if root_t_index is not None else 0
    
    num_chunks = (volume_num + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * chunk_size
        end_t = min(start_t + chunk_size, volume_num)
        chunk_vol_num = end_t - start_t
        
        # 创建子文件夹路径
        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # 为当前chunk创建metadata
        metadata = {
            "shape_t": chunk_vol_num + root_offset,
            "shape_c": 1,
            "shape_z": depth,
            "shape_y": height,
            "shape_x": width,
            "dtype": "uint8",
            # "global_t_start": start_t,  # 添加全局时间范围信息
            # "global_t_end": end_t - 1
        }
        if root_t_index is not None:
            metadata["root_t_index"] = int(root_t_index)
        
        metadata_path = os.path.join(subfolder_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if root_t_index is not None:
            print(f'Created {subfolder_name}/metadata.json with root_t={root_t_index} (local t=0 to {chunk_vol_num - 1}, global t={start_t} to {end_t-1})')
        else:
            print(f'Created {subfolder_name}/metadata.json (local t=0 to {chunk_vol_num-1}, global t={start_t} to {end_t-1})')
    
    if root_t_index is not None:
        print(f'Total: Created {num_chunks} metadata files with shared root_t={root_t_index}')
    else:
        print(f'Total: Created {num_chunks} metadata files')

def merge_chunked_annotations(zephir_folder_path, output_path=None, root_t_index=None, **kwargs):
    """
    合并所有chunk子文件夹中的annotations.h5文件为一个完整的annotations.h5
    使用abs_t_idx作为t_idx，从而可以用load_neuron_pt_tuple_from_annotations转换
    
    Parameters:
    zephir_folder_path: ZephIR文件夹路径，包含所有vol_*子文件夹
    output_path: 输出合并后的annotations.h5路径，默认为zephir_folder_path/annotations_merged.h5
    root_t_index: 如果提供，将在合并时跳过root_t_index对应的帧
    kwargs: 其他参数
    
    Returns:
    str: 合并后的annotations.h5文件路径
    """
    if root_t_index is None and 'root_t_index' in kwargs:
        root_t_index = kwargs.get('root_t_index')
    if root_t_index is not None:
        root_t_index = int(root_t_index)
    if output_path is None:
        output_path = os.path.join(zephir_folder_path, 'annotations_merged.h5')
    
    # 查找所有vol_*子文件夹
    def parse_chunk_range(name):
        match = re.match(r"vol_(\d+)_(\d+)$", name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None

    subfolders = []
    chunk_ranges = {}
    for item in os.listdir(zephir_folder_path):
        item_path = os.path.join(zephir_folder_path, item)
        if os.path.isdir(item_path) and item.startswith('vol_'):
            subfolders.append(item)
            chunk_ranges[item] = parse_chunk_range(item)

    # 按照起始时间排序子文件夹
    subfolders.sort(key=lambda name: chunk_ranges[name][0] if chunk_ranges.get(name) else float('inf'))
    
    if not subfolders:
        print(f"No vol_* subfolders found in {zephir_folder_path}")
        return None
    
    print(f"Found {len(subfolders)} chunks to merge: {subfolders}")

    metadata_cache = {}
    for subfolder in subfolders:
        metadata_path = os.path.join(zephir_folder_path, subfolder, 'metadata.json')
        if not os.path.exists(metadata_path):
            continue
        try:
            with open(metadata_path, 'r') as meta_file:
                metadata = json.load(meta_file)
            metadata_cache[subfolder] = metadata
            if root_t_index is None and 'root_t_index' in metadata:
                root_t_index = int(metadata['root_t_index'])
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    
    # 删除已存在的合并文件
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f'Deleted existing {output_path}')
    
    # 创建合并后的文件
    try:
        with h5py.File(output_path, 'a') as f_out:
            # 创建数据集
            max_shape = (None,)
            f_out.create_dataset('/x', shape=(0,), maxshape=max_shape, dtype='float32')
            f_out.create_dataset('/y', shape=(0,), maxshape=max_shape, dtype='float32')
            f_out.create_dataset('/z', shape=(0,), maxshape=max_shape, dtype='float32')
            f_out.create_dataset('/id', shape=(0,), maxshape=max_shape, dtype='uint32')
            idx_dtype = 'uint32'
            f_out.create_dataset('/parent_id', shape=(0,), maxshape=max_shape, dtype='uint32')
            f_out.create_dataset('/worldline_id', shape=(0,), maxshape=max_shape, dtype='uint8')
            f_out.create_dataset('/provenance', shape=(0,), maxshape=max_shape, dtype='S4')
            f_out.create_dataset('/t_idx', shape=(0,), maxshape=max_shape, dtype=idx_dtype)
            f_out.create_dataset('/abs_t_idx', shape=(0,), maxshape=max_shape, dtype=idx_dtype)
            
            total_points = 0
            global_id_offset = 0
            
            # 逐个读取并合并每个chunk的annotations
            removed_root_points = 0
            for subfolder in tqdm(subfolders, desc="Merging chunks"):
                annotation_path = os.path.join(zephir_folder_path, subfolder, 'annotations.h5')
                
                if not os.path.exists(annotation_path):
                    print(f"Warning: {annotation_path} not found, skipping")
                    continue
                
                chunk_range = chunk_ranges.get(subfolder)
                chunk_start = chunk_range[0] if chunk_range else None

                with h5py.File(annotation_path, 'r') as f_in:
                    # 检查是否有abs_t_idx
                    local_idx = f_in['/t_idx'][:].astype(np.int64)
                    abs_idx_in = f_in['/abs_t_idx'][:].astype(np.int64) if '/abs_t_idx' in f_in else None

                    chunk_metadata = metadata_cache.get(subfolder, {})
                    chunk_root_idx = chunk_metadata.get('root_t_index', root_t_index)
                    if chunk_root_idx is not None:
                        chunk_root_idx = int(chunk_root_idx)
                        if root_t_index is None:
                            root_t_index = chunk_root_idx

                    if chunk_root_idx is None and abs_idx_in is not None and np.any(local_idx == 0):
                        chunk_root_idx = int(np.unique(abs_idx_in[local_idx == 0])[0])
                        if root_t_index is None:
                            root_t_index = chunk_root_idx

                    root_present = np.any(local_idx == 0)

                    root_offset = 1 if root_present else 0
                    abs_idx_candidate = None
                    if chunk_start is not None:
                        abs_idx_candidate = local_idx - root_offset + chunk_start
                        if root_present and chunk_root_idx is not None:
                            abs_idx_candidate[local_idx == 0] = chunk_root_idx

                    def has_constant_offset(idx_local, idx_abs):
                        if idx_abs is None:
                            return False
                        if idx_local.size == 0:
                            return True
                        non_root_mask = idx_local != 0 if np.any(idx_local == 0) else np.ones_like(idx_local, dtype=bool)
                        if not np.any(non_root_mask):
                            return True
                        sample_indices = np.where(non_root_mask)[0][:500]
                        if sample_indices.size == 0:
                            return True
                        diffs = idx_abs[sample_indices] - idx_local[sample_indices]
                        return np.unique(diffs).size == 1

                    abs_idx_final = None
                    if has_constant_offset(local_idx, abs_idx_candidate):
                        abs_idx_final = abs_idx_candidate
                    elif has_constant_offset(local_idx, abs_idx_in):
                        abs_idx_final = abs_idx_in
                    else:
                        if abs_idx_candidate is not None:
                            abs_idx_final = abs_idx_candidate
                        elif abs_idx_in is not None:
                            abs_idx_final = abs_idx_in
                        else:
                            abs_idx_final = local_idx

                    if abs_idx_final is None:
                        print(f"Warning: Unable to determine absolute indices for {annotation_path}, skipping")
                        continue

                    keep_mask = np.ones_like(local_idx, dtype=bool)
                    if root_present:
                        keep_mask &= local_idx != 0
                        removed_root_points += np.sum(local_idx == 0)

                    if not np.any(keep_mask):
                        continue

                    abs_idx_filtered = abs_idx_final[keep_mask]

                    n = abs_idx_filtered.shape[0]
                    if n == 0:
                        continue

                    new_size = f_out['/x'].shape[0] + n

                    x_data = f_in['/x'][:][keep_mask]
                    f_out['/x'].resize(new_size, axis=0)
                    f_out['/x'][-n:] = x_data

                    y_data = f_in['/y'][:][keep_mask]
                    f_out['/y'].resize(new_size, axis=0)
                    f_out['/y'][-n:] = y_data

                    z_data = f_in['/z'][:][keep_mask]
                    f_out['/z'].resize(new_size, axis=0)
                    f_out['/z'][-n:] = z_data

                    worldline_data = f_in['/worldline_id'][:][keep_mask]
                    f_out['/worldline_id'].resize(new_size, axis=0)
                    f_out['/worldline_id'][-n:] = worldline_data

                    provenance_data = f_in['/provenance'][:][keep_mask]
                    f_out['/provenance'].resize(new_size, axis=0)
                    f_out['/provenance'][-n:] = provenance_data

                    abs_idx_filtered_uint32 = abs_idx_filtered.astype(np.uint32, copy=False)
                    f_out['/t_idx'].resize(new_size, axis=0)
                    f_out['/t_idx'][-n:] = abs_idx_filtered_uint32
                    f_out['/abs_t_idx'].resize(new_size, axis=0)
                    f_out['/abs_t_idx'][-n:] = abs_idx_filtered_uint32

                    new_id_values = np.arange(global_id_offset, global_id_offset + n, dtype=np.uint32)
                    f_out['/id'].resize(new_size, axis=0)
                    f_out['/id'][-n:] = new_id_values + 1

                    f_out['/parent_id'].resize(new_size, axis=0)
                    f_out['/parent_id'][-n:] = abs_idx_filtered_uint32 + 1

                    total_points += n
                    global_id_offset += n
            
            if root_t_index is not None:
                print(f'Successfully merged {len(subfolders)} chunks with {total_points} annotation points (root_t={root_t_index} removed, removed root points: {removed_root_points})')
            else:
                print(f'Successfully merged {len(subfolders)} chunks with {total_points} total annotation points (removed root points: {removed_root_points})')
            print(f'Merged file saved to: {output_path}')
    
    except Exception as e:
        print(f"Error merging annotations: {e}")
        return None
    
    return output_path

#%%
def convert_cbmi_data_to_ZephIR_format(infer_result_path, mat_folder_path, zephir_folder, **params):
    """
    将CBMI数据转换为ZephIR格式，数据按chunk分批保存
    
    Parameters:
    infer_result_path: 推理结果文件路径
    mat_folder_path: MAT文件夹路径
    zephir_folder: ZephIR输出文件夹路径
    params: 其他参数，包括chunk_size（默认100）
    """
    neuron_pt_tuple, volume_number = load_neuron_pt_tuple(infer_result_path)
    root_t_index = params.get('root_t_index')
    params_without_root = {k: v for k, v in params.items() if k != 'root_t_index'}
    volume_number = create_zephir_data(mat_folder_path, zephir_folder, volume_number, root_t_index=root_t_index, **params_without_root)
    create_zephir_annotations(neuron_pt_tuple, zephir_folder, root_t_index=root_t_index, **params_without_root)
    create_metadata_json(zephir_folder, volume_number, root_t_index=root_t_index, **params_without_root)


#%%
def load_neuron_pt_tuple_from_annotations(annotations_path, template_neuron_pt_tuple=None, **kwargs):
    """Load ZephIR annotations back into a ``neuron_pt_tuple`` tensor.

    This mirrors :func:`create_zephir_annotations` by using ``/t_idx`` as the
    absolute frame index and filling coordinates in the original order
    (sorted by ``worldline_id`` per frame). Optional template data can be used
    to restore the remaining feature columns (indices 3-7).

    Parameters
    ----------
    annotations_path : str
        Path to ``annotations.h5`` or the merged annotations file.
    template_neuron_pt_tuple : numpy.ndarray, optional
        Pre-loaded template tensor used to fill columns beyond XYZ.
    template_neuron_pt_tuple_path : str, optional (kwarg)
        Path to ``.npy``/``.h5`` file containing a reference
        ``neuron_pt_tuple``. When provided, columns 3-7 for overlapping
        frames/neurons are copied from the template.
    width, height, depth, z_ratio : numbers, optional (kwargs)
        Spatial scaling parameters used during export. Defaults match
        :func:`create_zephir_annotations`.

    Returns
    -------
    numpy.ndarray
        Array shaped ``(time, neurons, features)`` with at least 8 features
        (XYZ plus five placeholder columns). Returns ``None`` on failure.
    """

    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    z_ratio = kwargs.get('z_ratio', 5)
    template_path = kwargs.get('template_neuron_pt_tuple_path')

    def load_template(path):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.npy':
                return np.load(path)
            if ext == '.h5':
                with h5py.File(path, 'r') as template_file:
                    if 'neuron_pt_tuple' in template_file:
                        return template_file['neuron_pt_tuple'][:]
                    raise KeyError("'neuron_pt_tuple' dataset not found in template h5")
        except Exception as err:
            print(f"Failed to load template '{path}': {err}")
        return None

    template_array = None
    if template_neuron_pt_tuple is not None:
        template_array = np.asarray(template_neuron_pt_tuple, dtype=np.float32)
    elif template_path:
        if os.path.exists(template_path):
            template_array = load_template(template_path)
            if template_array is not None:
                template_array = np.asarray(template_array, dtype=np.float32)
        else:
            print(f"Template path '{template_path}' does not exist; skipping template merge")

    try:
        with h5py.File(annotations_path, 'r') as f:
            required_keys = ['/x', '/y', '/z', '/t_idx', '/worldline_id', '/provenance']
            for key in required_keys:
                if key not in f:
                    print(f"Missing dataset {key} in {annotations_path}")
                    return None

            x_rel = f['/x'][:]
            y_rel = f['/y'][:]
            z_rel = f['/z'][:]
            t_idx = f['/t_idx'][:].astype(np.int64)
            worldline_id = f['/worldline_id'][:]

            if x_rel.size == 0:
                print("Annotations file contains no points; returning empty tensor")
                return np.zeros((0, 0, 8), dtype=np.float32)

            if not (x_rel.shape == y_rel.shape == z_rel.shape == t_idx.shape == worldline_id.shape):
                print("Dataset length mismatch inside annotations file")
                return None

            if '/abs_t_idx' in f:
                abs_idx = f['/abs_t_idx'][:]
                if not np.array_equal(abs_idx, t_idx):
                    print("Warning: /abs_t_idx differs from /t_idx; using /t_idx only")

            x_abs = x_rel * width
            y_abs = y_rel * height
            z_abs = z_rel * (z_ratio * depth)

            unique_times = np.unique(t_idx)
            max_time_index = int(unique_times[-1]) + 1
            counts = np.bincount(t_idx, minlength=max_time_index)
            max_neurons = int(counts.max()) if counts.size else 0

            if template_array is not None:
                if template_array.ndim != 3 or template_array.shape[2] < 3:
                    print("Template tensor must have shape (time, neurons, features) with at least 3 features; ignoring template")
                    template_array = None
                    feature_dim = 8
                else:
                    feature_dim = max(8, template_array.shape[2])
            else:
                feature_dim = 8

            neuron_pt_tuple = np.zeros((max_time_index, max_neurons, feature_dim), dtype=np.float32)

            for t in unique_times:
                mask = (t_idx == t)
                indices = np.where(mask)[0]
                if indices.size == 0:
                    continue
                sort_order = np.argsort(worldline_id[indices])
                ordered_idx = indices[sort_order]

                n_neurons = ordered_idx.size
                neuron_pt_tuple[t, :n_neurons, 0] = x_abs[ordered_idx]
                neuron_pt_tuple[t, :n_neurons, 1] = y_abs[ordered_idx]
                neuron_pt_tuple[t, :n_neurons, 2] = z_abs[ordered_idx]

                if template_array is not None and t < template_array.shape[0]:
                    template_frame = template_array[t]
                    copy_neurons = min(n_neurons, template_frame.shape[0])
                    if copy_neurons > 0 and template_frame.shape[1] >= 3:
                        neuron_pt_tuple[t, :copy_neurons, 3:feature_dim] = template_frame[:copy_neurons, 3:feature_dim]

            print(f"Successfully loaded {unique_times.size} timepoints with up to {max_neurons} neurons per frame")
            return neuron_pt_tuple

    except Exception as e:
        print(f"Error loading annotations from {annotations_path}: {e}")
        return None


def convert_annotations_to_neuron_pt_tuple(
    annotations_path,
    output_path,
    template_neuron_pt_tuple=None,
    template_neuron_pt_tuple_path=None,
    **kwargs,
):
    """
    将ZephIR格式的annotations.h5转换为neuron_pt_tuple格式并保存
    
    Parameters:
    annotations_path: 输入的annotations.h5文件路径
    output_path: 输出的neuron_pt_tuple文件路径 (.npy或.h5)
    template_neuron_pt_tuple: 已加载的neuron_pt_tuple模板，用于填充第3-7维数据
    template_neuron_pt_tuple_path: 模板文件路径(.npy或.h5)，与上者二选一即可
    kwargs: 包含width, height, depth, z_ratio等参数
    """
    if template_neuron_pt_tuple_path is not None:
        kwargs.setdefault('template_neuron_pt_tuple_path', template_neuron_pt_tuple_path)

    neuron_pt_tuple = load_neuron_pt_tuple_from_annotations(
        annotations_path,
        template_neuron_pt_tuple=template_neuron_pt_tuple,
        **kwargs,
    )
    
    if neuron_pt_tuple is None:
        print("Failed to convert annotations")
        return

    if output_path.endswith('.npy'):
        np.save(output_path, neuron_pt_tuple)
        print(f"Saved neuron_pt_tuple to {output_path}")
    elif output_path.endswith('.h5'):
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('neuron_pt_tuple', data=neuron_pt_tuple)
        print(f"Saved neuron_pt_tuple to {output_path}")
    else:
        raise ValueError("Output file must be .npy or .h5 format")

# %%
if __name__ == "__main__":
    infer_result_path = r"Y:\20251106_w5_proxy\dynamics.h5"
    zephir_folder =  r'I:\WJH\infer\manual\registration_annotation\20251106\w5\zephir_data'
    mat_folder_path = r"Z:\data4\Ikrma\20251106\w5_AWCON\w5\red"
    xoy_unit = 0.3
    z_unit = 1.5
    z_ratio = z_unit / xoy_unit
    params = dict(
        z_ratio = z_ratio,
        width = 1024,
        height = 1024,
        depth = 18,
        volume_shape = (1024, 1024, 18),
        denoise_range = (120, 1000),
        root_t_index = 134,
        chunk_size = 100,  # 每100个时间帧保存为一个文件
    )

    convert_cbmi_data_to_ZephIR_format(infer_result_path, mat_folder_path, zephir_folder, **params)
    
    # merged_annotations_path = merge_chunked_annotations(zephir_folder, **params)

    # # # # if merged_annotations_path:
    # output_npy_path = os.path.join(zephir_folder, 'neuron_pt_tuple.npy')
    # convert_annotations_to_neuron_pt_tuple(merged_annotations_path, output_npy_path,template_neuron_pt_tuple_path=r'I:\WJH\infer\manual\registration_annotation\20250730\w3_freelymoving\neuron_pt_tuple.npy', **params)