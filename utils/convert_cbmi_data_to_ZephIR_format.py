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


def create_zephir_data(mat_folder_path, zephir_folder, vol_num, **kwargs):
    """
    从包含.mat文件的文件夹中读取cell array数据，转换为ZephIR格式的data.h5
    
    Parameters:
    mat_folder_path: 包含.mat文件的文件夹路径
    zephir_folder: ZephIR输出文件夹路径
    vol_num: 处理的时间帧数量
    kwargs: 其他参数，包括volume_shape, denoise_range等
    
    Returns:
    int: 处理的时间帧数量
    """
    # 从kwargs中提取参数
    volume_shape = kwargs.get('volume_shape', (1024, 1024, 18))
    denoise_range = kwargs.get('denoise_range', (120, 1000))
        
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
    
    # 将所有体数据组合成ZephIR格式
    vol_num_mat = len(all_volumes)
    if vol_num_mat != vol_num:
        vol_num = min(vol_num_mat, vol_num)
    slice_num = all_volumes[0].shape[0]  # z轴切片数
    height, width = all_volumes[0].shape[1], all_volumes[0].shape[2]
    
    # 创建ZephIR格式的数据数组: (time, channel, z, y, x)
    img_data = np.empty((vol_num, 1, slice_num, height, width), dtype=np.uint8)
    
    for i, volume in enumerate(all_volumes[:vol_num]):
        img_data[i, 0] = volume
    
    # 保存数据
    os.makedirs(zephir_folder, exist_ok=True)
    data_path = os.path.join(zephir_folder, 'data.h5')
    with h5py.File(data_path, 'w') as hf:
        hf.create_dataset('data', data=img_data, chunks=True)
    
    print(f"Successfully created data.h5 with {vol_num} timepoints")
    return vol_num
    

def create_zephir_annotations(neuron_pt_tuple, zephir_folder_path, **kwargs):
    '''
    input: all neuron_pt_tuple (time x neuron_pt_tuple)
    output: annotations.h5
    '''
    # get parameter
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    z_ratio = kwargs.get('z_ratio', 5)

    # copy from Rong's code
    h5file_path = os.path.join(zephir_folder_path, 'annotations.h5')
    l = len(neuron_pt_tuple)
    if os.path.exists(h5file_path):
        os.remove(h5file_path)
        print('deleting the existing annotations.h5')
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

            for volume_index, neuron_pt_tuple in enumerate(neuron_pt_tuple):
                n = neuron_pt_tuple.shape[0]
                new_size = f['/x'].shape[0] + n
                f['/x'].resize(new_size, axis=0)
                f['/x'][-n:] = neuron_pt_tuple[:, 0] / width
                f['/y'].resize(new_size, axis=0)
                f['/y'][-n:] = neuron_pt_tuple[:, 1] / height
                f['/z'].resize(new_size, axis=0)
                f['/z'][-n:] = neuron_pt_tuple[:, 2] / (z_ratio * depth) # z-axis normalization factor: frame_num * (z_unit / XOY_unit)
                # f['/id'].resize(new_size, axis=0)
                # f['/id'][-n:] = np.zeros(n)
                f['/worldline_id'].resize(new_size, axis=0)
                f['/worldline_id'][-n:] = np.arange(0, n)
                f['/provenance'].resize(new_size, axis=0)
                f['/provenance'][-n:] = np.array(['ANTT']*n, dtype='S4')
                f['/t_idx'].resize(new_size, axis=0)
                f['/t_idx'][-n:] = np.full(n, volume_index, dtype='uint16')

                # 计算当前全局起始索引（即之前已写入的数据总量）
                current_start = new_size - n  # 等同于 f['/id'].shape[0] - n
                # 生成从 current_start 开始的连续整数序列
                id_values = np.arange(current_start, current_start + n, dtype='uint32')
                # 追加到 id 和 parent_id 数据集
                f['/id'].resize(new_size, axis=0)
                f['/id'][-n:] = id_values + 1
                f['/parent_id'].resize(new_size, axis=0)
                f['/parent_id'][-n:] = np.full(n, volume_index+1, dtype='uint16')  # parent_id 与 t相同
                
            print('finish creating new annotations.h5')

    except Exception as e:
        print("An error occurred:", e)
        return None

    return None


import json
def create_metadata_json(zephir_folder_path, volume_num, **kwargs):
    width = kwargs.get('width', 1024)
    height = kwargs.get('height', 1024)
    depth = kwargs.get('depth', 18)
    metadata = {"shape_t": volume_num, "shape_c": 1, "shape_z": depth, "shape_y": height, "shape_x": width, "dtype": "uint8"}
    with open(os.path.join(zephir_folder_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

#%%
def convert_cbmi_data_to_ZephIR_format(infer_result_path, mat_folder_path, zephir_folder, **params):
    neuron_pt_tuple, volume_number = load_neuron_pt_tuple(infer_result_path)
    # volume_number = create_zephir_data(mat_folder_path, zephir_folder, volume_number, **params)
    create_zephir_annotations(neuron_pt_tuple[:volume_number], zephir_folder, **params)
    create_metadata_json(zephir_folder, volume_number, **params)


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
            t_idx = f['/t_idx'][:]  # 时间索引
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
            
            max_neurons = max(max_neurons_per_frame) if max_neurons_per_frame else 0
            
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
    
    if neuron_pt_tuple is not None:
        if output_path.endswith('.npy'):
            np.save(output_path, neuron_pt_tuple)
            print(f"Saved neuron_pt_tuple to {output_path}")
        elif output_path.endswith('.h5'):
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('neuron_pt_tuple', data=neuron_pt_tuple)
            print(f"Saved neuron_pt_tuple to {output_path}")
        else:
            raise ValueError("Output file must be .npy or .h5 format")
    else:
        print("Failed to convert annotations")

# %%
if __name__ == "__main__":
    infer_result_path = r"I:\WJH\infer\runhui_new\dynamics.h5"
    zephir_folder =  r'I:\WJH\infer\runhui_new\ZephIR'
    mat_folder_path = r"I:\WJH\infer\runhui_new\mat"
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
    )

    convert_cbmi_data_to_ZephIR_format(infer_result_path, mat_folder_path, zephir_folder, **params)