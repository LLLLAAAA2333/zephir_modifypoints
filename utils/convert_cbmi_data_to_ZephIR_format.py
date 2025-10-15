# %%
# import packages
import sys
import h5py
import numpy as np
import os
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.MatToolkit import extract_number_from_filename, get_matlab_file_info, load_single_timepoint_from_matlab
from utils.logged_operation import logged_operation
from tqdm import tqdm
#%%
def load_neuron_pt_tuple(infer_result):
    with h5py.File(infer_result, 'r') as f:
        all_groups = [k for k in f.keys()]
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
    image_float32 = image.astype(np.float32)
    # Check that source_min is less than source_max
    if source_min >= source_max:
        raise ValueError("source_min must be less than source_max")
    # Compute the rescaled image with values adjusted to the new range and clip to ensure
    # values stay within target_min and target_max
    rescaled_image = np.clip((image_float32 - source_min) / (source_max - source_min) * (target_max - target_min) + target_min,
                             target_min, target_max).astype(image.dtype)
    return rescaled_image


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
    
    # 收集所有时间点的数据
    all_volumes = []
    total_timepoints = 0
    
    # 计算总的时间点数
    for mat_file in mat_files:
        filepath = os.path.join(mat_folder_path, mat_file)
        file_info = get_matlab_file_info(filepath)
        if file_info and file_info['is_cell_array']:
            total_timepoints += file_info['shape'][1]
        elif file_info:
            total_timepoints += 1
    
    print(f"Total timepoints to process: {total_timepoints}")
    
    # 处理每个.mat文件
    for i, mat_file in enumerate(tqdm(mat_files)):
        with logged_operation(display_context=f"Processing {mat_file} ({i+1}/{len(mat_files)})"):
            filepath = os.path.join(mat_folder_path, mat_file)
            file_info = get_matlab_file_info(filepath)
            
            if file_info is None:
                print(f"Warning: Could not get info for {mat_file}")
                continue
            
            # 确定时间点数量
            if file_info['is_cell_array']:
                num_timepoints = file_info['shape'][1]
            else:
                num_timepoints = 1
            
            # 读取每个时间点的数据
            for timepoint_idx in tqdm(range(num_timepoints),desc="Processing"):
                volume = load_single_timepoint_from_matlab(filepath, timepoint_idx)
                
                if volume is not None and volume.size > 0:
                    # 转置数据以匹配期望的形状 (z, x, y) -> (z, y, x)
                    volume = np.transpose(volume, (0, 2, 1))
                    
                    # 去噪处理
                    if denoise_range:
                        volume[(volume < denoise_range[0]) | (volume > denoise_range[1])] = 0
                    
                    # 缩放到uint8范围
                    volume_scaled = rescale_image(volume, 0, 255).astype(np.uint8)
                    all_volumes.append(volume_scaled)
                else:
                    print(f"Warning: No valid data at timepoint {timepoint_idx} in {mat_file}")
    
    if not all_volumes:
        raise ValueError("No valid volume data found in any .mat files")
    
    # 确定最终处理的数据量
    vol_num_mat = len(all_volumes)
    if vol_num_mat != vol_num:
        vol_num = min(vol_num_mat, vol_num)
    root_volume = None
    if root_t_index is not None:
        root_t_index = int(root_t_index)
        if not (0 <= root_t_index < vol_num_mat):
            raise ValueError(f"root_t_index {root_t_index} is out of range for {vol_num_mat} volumes")
        root_volume = all_volumes[root_t_index]
    slice_num = all_volumes[0].shape[0]  # z轴切片数
    height, width = all_volumes[0].shape[1], all_volumes[0].shape[2]
    
    # 创建主文件夹
    os.makedirs(zephir_folder, exist_ok=True)
    
    # 按chunk_size分批保存数据
    num_chunks = (vol_num + chunk_size - 1) // chunk_size  # 向上取整
    
    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * chunk_size
        end_t = min(start_t + chunk_size, vol_num)
        
        # 创建子文件夹名称，例如 vol_0_99, vol_100_199
        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # 当前chunk的时间帧数量
        chunk_vol_num = end_t - start_t
        root_offset = 1 if root_volume is not None else 0
        total_frames = chunk_vol_num + root_offset
        
        # 创建当前chunk的数据数组: (time, channel, z, y, x)
        img_data = np.empty((total_frames, 1, slice_num, height, width), dtype=np.uint8)
        frame_offset = 0
        if root_volume is not None:
            img_data[0, 0] = root_volume
            frame_offset = 1
        
        for i in range(chunk_vol_num):
            img_data[frame_offset + i, 0] = all_volumes[start_t + i]
        
        # 保存当前chunk的数据
        data_path = os.path.join(subfolder_path, 'data.h5')
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset('data', data=img_data, chunks=True)
        
        if root_volume is not None:
            print(f"Successfully created {subfolder_name}/data.h5 with {chunk_vol_num} timepoints plus root (t={start_t} to {end_t-1}, root_t={root_t_index})")
        else:
            print(f"Successfully created {subfolder_name}/data.h5 with {chunk_vol_num} timepoints (t={start_t} to {end_t-1})")
    
    if root_volume is not None:
        print(f"Total: Created {num_chunks} chunks with {vol_num} timepoints plus shared root")
    else:
        print(f"Total: Created {num_chunks} chunks with {vol_num} timepoints")
    return vol_num
    

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
    subfolders = []
    for item in os.listdir(zephir_folder_path):
        item_path = os.path.join(zephir_folder_path, item)
        if os.path.isdir(item_path) and item.startswith('vol_'):
            subfolders.append(item)
    
    # 按照起始时间排序子文件夹
    def extract_start_t(folder_name):
        # 从 "vol_0_99" 中提取起始时间 0
        parts = folder_name.split('_')
        return int(parts[1])
    
    subfolders.sort(key=extract_start_t)
    
    if not subfolders:
        print(f"No vol_* subfolders found in {zephir_folder_path}")
        return None
    
    print(f"Found {len(subfolders)} chunks to merge: {subfolders}")

    if root_t_index is None:
        for subfolder in subfolders:
            metadata_path = os.path.join(zephir_folder_path, subfolder, 'metadata.json')
            if not os.path.exists(metadata_path):
                continue
            try:
                with open(metadata_path, 'r') as meta_file:
                    metadata = json.load(meta_file)
                if 'root_t_index' in metadata:
                    root_t_index = int(metadata['root_t_index'])
                    break
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
            f_out.create_dataset('/parent_id', shape=(0,), maxshape=max_shape, dtype='uint16')
            f_out.create_dataset('/worldline_id', shape=(0,), maxshape=max_shape, dtype='uint8')
            f_out.create_dataset('/provenance', shape=(0,), maxshape=max_shape, dtype='S4')
            f_out.create_dataset('/t_idx', shape=(0,), maxshape=max_shape, dtype='uint16')
            
            total_points = 0
            global_id_offset = 0
            
            # 逐个读取并合并每个chunk的annotations
            for subfolder in tqdm(subfolders, desc="Merging chunks"):
                annotation_path = os.path.join(zephir_folder_path, subfolder, 'annotations.h5')
                
                if not os.path.exists(annotation_path):
                    print(f"Warning: {annotation_path} not found, skipping")
                    continue
                
                with h5py.File(annotation_path, 'r') as f_in:
                    # 检查是否有abs_t_idx
                    if '/abs_t_idx' not in f_in:
                        print(f"Warning: {annotation_path} does not contain /abs_t_idx, skipping")
                        continue
                    
                    abs_idx_in = f_in['/abs_t_idx'][:]
                    mask_array = None
                    if root_t_index is not None:
                        mask_array = abs_idx_in != root_t_index
                        if mask_array.ndim != 1:
                            mask_array = None

                    n = np.count_nonzero(mask_array) if mask_array is not None else f_in['/x'].shape[0]
                    if n == 0:
                        continue
                    
                    new_size = f_out['/x'].shape[0] + n
                    
                    # 复制所有数据
                    x_data = f_in['/x'][:]
                    if mask_array is not None:
                        x_data = x_data[mask_array]
                    f_out['/x'].resize(new_size, axis=0)
                    f_out['/x'][-n:] = x_data

                    y_data = f_in['/y'][:]
                    if mask_array is not None:
                        y_data = y_data[mask_array]
                    f_out['/y'].resize(new_size, axis=0)
                    f_out['/y'][-n:] = y_data

                    z_data = f_in['/z'][:]
                    if mask_array is not None:
                        z_data = z_data[mask_array]
                    f_out['/z'].resize(new_size, axis=0)
                    f_out['/z'][-n:] = z_data

                    worldline_data = f_in['/worldline_id'][:]
                    if mask_array is not None:
                        worldline_data = worldline_data[mask_array]
                    f_out['/worldline_id'].resize(new_size, axis=0)
                    f_out['/worldline_id'][-n:] = worldline_data

                    provenance_data = f_in['/provenance'][:]
                    if mask_array is not None:
                        provenance_data = provenance_data[mask_array]
                    f_out['/provenance'].resize(new_size, axis=0)
                    f_out['/provenance'][-n:] = provenance_data

                    abs_idx_filtered = abs_idx_in if mask_array is None else abs_idx_in[mask_array]
                    f_out['/t_idx'].resize(new_size, axis=0)
                    f_out['/t_idx'][-n:] = abs_idx_filtered
                    
                    # 重新生成全局唯一的id
                    new_id_values = np.arange(global_id_offset, global_id_offset + n, dtype='uint32')
                    f_out['/id'].resize(new_size, axis=0)
                    f_out['/id'][-n:] = new_id_values + 1
                    
                    # parent_id使用abs_t_idx + 1
                    f_out['/parent_id'].resize(new_size, axis=0)
                    f_out['/parent_id'][-n:] = abs_idx_filtered + 1
                    
                    total_points += n
                    global_id_offset += n
            
            if root_t_index is not None:
                print(f'Successfully merged {len(subfolders)} chunks with {total_points} annotation points (root_t={root_t_index} removed)')
            else:
                print(f'Successfully merged {len(subfolders)} chunks with {total_points} total annotation points')
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
def load_neuron_pt_tuple_from_annotations(annotations_path, **kwargs):
    """
    从ZephIR格式的annotations.h5文件中读取相对坐标并转换回neuron_pt_tuple格式
    这是create_zephir_annotations的反函数
    
    Parameters:
    annotations_path: annotations.h5文件路径
    kwargs: 包含width, height, depth, z_ratio等参数
    
    Returns:
    numpy.ndarray: neuron_pt_tuple格式的数据 (time, neurons, coordinates)
                   coordinates的前3维为(x, y, z)绝对坐标
    """
    # 获取参数
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024) 
    depth = kwargs.get('depth', 18)
    z_ratio = kwargs.get('z_ratio', 5)
    
    try:
        with h5py.File(annotations_path, 'r') as f:
            # 读取所有数据
            x_rel = f['/x'][:]  # 相对x坐标 (0-1)
            y_rel = f['/y'][:]  # 相对y坐标 (0-1)
            z_rel = f['/z'][:]  # 相对z坐标 (0-1)
            
            # 优先使用abs_t_idx（绝对时间索引），如果不存在则使用t_idx
            if '/abs_t_idx' in f:
                t_idx = f['/abs_t_idx'][:]  # 使用绝对时间索引
                print("Using abs_t_idx for time indexing")
            else:
                t_idx = f['/t_idx'][:]  # 使用相对时间索引
                print("Using t_idx for time indexing (abs_t_idx not found)")
            
            worldline_id = f['/worldline_id'][:]  # 神经元ID
            
            # 转换回绝对坐标
            x_abs = x_rel * width
            y_abs = y_rel * height
            z_abs = z_rel * (z_ratio * depth)
            
            # 获取时间点数量和最大神经元数量
            max_t = int(np.max(t_idx)) + 1
            max_neurons_per_frame = []
            
            # 计算每个时间点的神经元数量
            for t in range(max_t):
                mask = (t_idx == t)
                if np.any(mask):
                    max_neurons_per_frame.append(np.sum(mask))
                else:
                    max_neurons_per_frame.append(0)
            
            max_neurons = max(max_neurons_per_frame, default=0)
            
            # 初始化neuron_pt_tuple数组 (time, neurons, 8)
            # 前3维是坐标，后面的维度用0填充或根据需要设置
            neuron_pt_tuple = np.zeros((max_t, max_neurons, 8), dtype=np.float32)
            
            # 按时间点组织数据
            for t in range(max_t):
                mask = (t_idx == t)
                if np.any(mask):
                    # 获取当前时间点的数据
                    t_x = x_abs[mask]
                    t_y = y_abs[mask]
                    t_z = z_abs[mask]
                    t_worldline = worldline_id[mask]
                    
                    # 按worldline_id排序以保持一致的顺序
                    sort_idx = np.argsort(t_worldline)
                    t_x = t_x[sort_idx]
                    t_y = t_y[sort_idx]
                    t_z = t_z[sort_idx]
                    
                    # 填入坐标数据
                    n_neurons = len(t_x)
                    neuron_pt_tuple[t, :n_neurons, 0] = t_x  # x坐标
                    neuron_pt_tuple[t, :n_neurons, 1] = t_y  # y坐标
                    neuron_pt_tuple[t, :n_neurons, 2] = t_z  # z坐标
                    # 第3-7维保持为0，可以根据需要修改
            
            print(f"Successfully loaded {max_t} timepoints with up to {max_neurons} neurons per frame")
            return neuron_pt_tuple
            
    except Exception as e:
        print(f"Error loading annotations from {annotations_path}: {e}")
        return None


def convert_annotations_to_neuron_pt_tuple(annotations_path, output_path, **kwargs):
    """
    将ZephIR格式的annotations.h5转换为neuron_pt_tuple格式并保存
    
    Parameters:
    annotations_path: 输入的annotations.h5文件路径
    output_path: 输出的neuron_pt_tuple文件路径 (.npy或.h5)
    kwargs: 包含width, height, depth, z_ratio等参数
    """
    neuron_pt_tuple = load_neuron_pt_tuple_from_annotations(annotations_path, **kwargs)
    
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
    infer_result_path = r"I:\WJH\infer\test\0730\w3\dynamics.h5"
    zephir_folder =  r'I:\WJH\infer\test\0730\w3\zephir'
    mat_folder_path = r"I:\WJH\infer\test\0730\w3\mat"
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
        root_t_index = 158,
        chunk_size = 100,  # 每100个时间帧保存为一个文件
    )

    convert_cbmi_data_to_ZephIR_format(infer_result_path, mat_folder_path, zephir_folder, **params)
    
    # 可选：合并所有chunk的annotations.h5文件
    # merged_annotations_path = merge_chunked_annotations(zephir_folder, **params)
    
    # 可选：将合并后的annotations.h5转换为neuron_pt_tuple格式
    # if merged_annotations_path:
    #     output_npy_path = os.path.join(zephir_folder, 'neuron_pt_tuple.npy')
    #     convert_annotations_to_neuron_pt_tuple(merged_annotations_path, output_npy_path, **params)