if __name__ == "__main__":
    input_folder = r"H:\Process_temporary\WJH\immobile_data\w3\ref"
    output_folder = r"H:\Process_temporary\WJH\immobile_data\w3\ref\test_refine_3"
    args = {
        "input_folder": input_folder,
        "output_folder": output_folder,
        "volume_shape": (1024, 1024, 18)
    }
#%%
import os
import re
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logged_operation import logged_operation
#%%
def extract_number_from_filename(filename):
    """从文件名中提取ImgStk后的数字用于排序"""
    img_match = re.search(r'ImgStk(\d+)', filename)
    dk_match = re.search(r'dk(\d+)', filename)
    img_num = int(img_match.group(1)) if img_match else 0
    dk_num = int(dk_match.group(1)) if dk_match else 0
    return (img_num, dk_num)

def get_matlab_file_info(filepath):
    """获取MATLAB v7.3文件的基本信息和数据键，用于延迟读取"""
    try:
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            
            # 通常第一个非'#'开头的键是主要数据
            data_key = None
            for key in keys:
                if not key.startswith('#'):
                    data_key = key
                    break
            
            if data_key is None:
                raise ValueError("No valid data key found")
            
            # 获取数据形状和类型信息
            cell_data = f[data_key]
            return {
                'data_key': data_key,
                'shape': cell_data.shape,
                'dtype': cell_data.dtype,
                'is_cell_array': cell_data.dtype == 'object'
            }
    except Exception as e:
        print(f"Error getting info from {filepath}: {e}")
        return None

def load_single_timepoint_from_matlab(filepath, timepoint_index):
    """从MATLAB v7.3文件中延迟读取单个时间点的数据"""
    try:
        with h5py.File(filepath, 'r') as f:
            # 获取数据键
            keys = list(f.keys())
            data_key = None
            for key in keys:
                if not key.startswith('#'):
                    data_key = key
                    break
            
            if data_key is None:
                return None
            
            cell_data = f[data_key]
            
            if cell_data.dtype == 'object':
                # 处理cell数组 - 延迟读取指定时间点
                if timepoint_index < cell_data.shape[1]:
                    ref = cell_data[0, timepoint_index]
                    if isinstance(ref, h5py.Reference):
                        array_data = f[ref][:]
                        return array_data
                return None
            else:
                # 直接是数组数据
                if timepoint_index == 0:
                    return cell_data[:]
                return None
                
    except Exception as e:
        print(f"Error reading timepoint {timepoint_index} from {filepath}: {e}")
        return None

def process_matlab_to_hdf5(input_folder, output_folder, volume_shape = (1024, 1024, 18)):
    """
    将MATLAB文件转换为HDF5格式，使用延迟读取避免内存溢出
    每个HDF5文件代表一个时间点
    
    Parameters:
    input_folder: 包含.mat文件的文件夹路径
    output_folder: 输出HDF5文件的文件夹路径
    volume_shape: 期望的体帧数据形状 (默认: (y, x, z) = (1024, 1024, 18))
    """

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    mat_files = [file for file in os.listdir(input_folder) if file.endswith('.mat') and 'ImgStk' in file]
    mat_files.sort(key=extract_number_from_filename)
    print(f"Found {len(mat_files)} MATLAB files")

    expected_shape = (volume_shape[2], volume_shape[0], volume_shape[1])  # (z, y, x)
    total_count = 0
    output_file_count = 0

    for i, mat_file in enumerate(tqdm(mat_files)):
        with logged_operation(display_context=f"Processing {mat_file} ({i+1}/{len(mat_files)})"):
            # print(f"Processing {mat_file} ({i+1}/{len(mat_files)})")
            filepath = os.path.join(input_folder, mat_file)
            
            try:
                # 获取文件信息
                file_info = get_matlab_file_info(filepath)
                if file_info is None:
                    print(f"Warning: Could not get info for {mat_file}")
                    continue
                
                # 确定这个文件中有多少个时间点
                if file_info['is_cell_array']:
                    num_timepoints = file_info['shape'][1]
                else:
                    num_timepoints = 1
                
                # 对每个时间点进行处理
                for timepoint_idx in tqdm(range(num_timepoints),desc="Saving"):
                    # with logged_operation(display_context=f"Processing {mat_file} - Timepoint {timepoint_idx+1}/{num_timepoints}"):
                        data_array = load_single_timepoint_from_matlab(filepath, timepoint_idx)
                        
                        if data_array is not None and data_array.size > 0:
                            # 转置数据以匹配期望的形状
                            data_array = np.transpose(data_array, (0, 2, 1))
                            
                            if data_array.shape == expected_shape:
                                output_filename = f"ImgStk_{output_file_count:06d}.h5"
                                output_path = os.path.join(output_folder, output_filename)
                                
                                try:
                                    with h5py.File(output_path, 'w') as hf:
                                        # 移除compression参数，使用默认的非压缩存储
                                        hf.create_dataset(
                                            'data', 
                                            data=data_array[np.newaxis, :, :, :],
                                            chunks=True
                                        )
                                    output_file_count += 1
                                    total_count += 1
                                except Exception as e:
                                    print(f"Error saving {output_filename}: {e}")
                            else:
                                print(f"Warning: Data shape {data_array.shape} != expected {expected_shape}")
                        else:
                            print(f"Warning: No valid data at timepoint {timepoint_idx} in {mat_file}")
                        
            except Exception as e:
                print(f"Error processing {mat_file}: {e}")
                continue

    print(f"Total valid arrays processed: {total_count}")
    print(f"Total HDF5 files created: {output_file_count}")

def load_converted_hdf5(h5_data_path):
    """
    加载转换后的HDF5文件并返回数据列表
    """
    with h5py.File(h5_data_path, 'r') as hf:
        image_data = hf['data'][:]
    return image_data

#%%
if __name__ == "__main__":
    process_matlab_to_hdf5(**args)