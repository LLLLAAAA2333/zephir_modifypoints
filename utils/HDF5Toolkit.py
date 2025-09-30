import h5py
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def save_to_hdf5(group, name, data):
    """
    Save data to an HDF5 group with type handling.
    :param group: HDF5 group to save data into.
    :param name: Name of the dataset or group to create.
    :param data: Data to save, can be various types including numpy arrays,
    """
    if type(data) == type(None) :
        data_type = "None"
    else:
        data_type = str(type(data)) 

    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        if data.dtype == torch.float16:
            data = data.to(torch.float32)
        dset = group.create_dataset(name, data=data.cpu().numpy())
    elif isinstance(data, np.ndarray):
        dset = group.create_dataset(name, data=data)
    elif isinstance(data, (np.integer, np.floating, np.complexfloating)):
        # Handle numpy scalar types
        dset = group.create_dataset(name, data=data)
    elif isinstance(data, (str, int, float, complex)):
        dset = group.create_dataset(name, data=data)
    elif isinstance(data, bytes):
        # Convert bytes to string for HDF5 compatibility
        dset = group.create_dataset(name, data=data.decode('utf-8'))
    elif data is None:
        dset = group.create_dataset(name, shape=(0,))
    elif isinstance(data, dict):
        dset = group.create_group(name)
        for key, value in data.items():
            save_to_hdf5(dset, str(key), value)
    elif isinstance(data, (list, tuple)):
        try:
            np_array_data = np.array(data)
            dset = group.create_dataset(name, data=np_array_data)
        except (ValueError, TypeError):
            dset = group.create_group(name)
            for idx, item in enumerate(data):
                save_to_hdf5(dset, str(idx), item)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    dset.attrs['data_type'] = data_type

def save_h5file(path, root_name, mode = "a",**kwargs):
    with h5py.File(path, mode) as file:
        for key, value in kwargs.items():
            save_to_hdf5(file, f"{root_name}/{key}", value)


def load_from_hdf5(item):
    """
    Recursively load data from an HDF5 dataset or group,
    reconstructing the original data structure.
    
    :param item: An h5py dataset or group
    :return: The reconstructed data
    """
    # Check if it's a group (has keys) or a dataset
    if hasattr(item, 'keys'):
        # It's a group
        if 'data_type' in item.attrs and ('list' in item.attrs['data_type'] or 'tuple' in item.attrs['data_type']):
            # It's a list or tuple that couldn't be stored as a dataset
            result = []
            # Sort indices to maintain original order
            indices = sorted([int(idx) for idx in item.keys()])
            for idx in indices:
                result.append(load_from_hdf5(item[str(idx)]))
            
            # Convert to tuple if original was tuple
            if 'tuple' in item.attrs['data_type']:
                result = tuple(result)
            return result
        else:
            # It's a dictionary
            result = {}
            for key in item.keys():
                result[key] = load_from_hdf5(item[key])
            return result
    else:
        # It's a dataset, load the value
        if len(item.shape) == 0:  # scalar
            data = item[()]
        else:
            data = item[:]
            
        # Check if original was None
        if 'data_type' in item.attrs:
            if 'NoneType' in item.attrs['data_type'] or item.attrs['data_type'] == 'None':
                return None
            
            # Optional: convert back to torch tensor if original was a tensor
            if TORCH_AVAILABLE and 'torch.Tensor' in item.attrs['data_type']:
                data = torch.tensor(data)
                
        return data

def load_h5file(path, root_name):
    """
    Load data from an HDF5 file that was saved with save_h5file.
    
    :param path: Path to the HDF5 file
    :param root_name: The root group name that was used when saving
    :return: Dictionary with loaded data
    """
    result = {}
    with h5py.File(path, 'r') as file:
        if root_name not in file:
            raise KeyError(f"Root name '{root_name}' not found in the HDF5 file")
            
        root_group = file[root_name]
        for key in root_group.keys():
            result[key] = load_from_hdf5(root_group[key])
            
    return result