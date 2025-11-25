import numpy as np
import h5py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
from utils.convert_cbmi_data_to_ZephIR_format import load_neuron_pt_tuple
from glob import glob

def parse_neuron_pt_tuple_name(filename):
    fname = os.path.basename(filename)
    match = re.match(r'^ImgStk(\d+)_dk(\d+)_w(\d+)_Dt(\d+?)_(\d+)\.npy$', fname)
    if not match:
        return None
    stk, dk, w, date, seq = map(int, match.groups())
    return {
        'stk': stk,
        'dk': int(dk),
        'w': int(w),
        'date': int(date),
        'seq': int(seq),
        'path': filename
    }
    
def sort_neuron_pt_tuple_file_list(neuron_pt_tuple_folder, pattern='ImgStk*_dk*_w*_Dt*_*.npy'):
    files = glob(os.path.join(neuron_pt_tuple_folder, pattern))
    parsed_files = [parse_neuron_pt_tuple_name(f) for f in files]
    parsed_files = [f for f in parsed_files if f is not None]
    sorted_files = sorted(parsed_files, key=lambda x: (x['date'], x['stk'], x['dk'], x['w'], x['seq']))
    return [f['path'] for f in sorted_files]

def neuron_pt_tuple2npz_pair(neuron_pt_tuple_hat, raw_neuron_pt_tuple_folder, npz_folder_path, root_idx, **kwargs):
    """
    Args:
        neuron_pt_tuple_hat: modified neuron points array with shape (T, N, 8)
        raw_neuron_pt_tuple_folder: including original neuron points array with shape (N, 8)
        npz_folder_path: folder path to save npz files
        root_idx: index of the root time point in the neuron points array
    """
    os.makedirs(npz_folder_path, exist_ok=True)
    pattern = r'(Dt)\d{6,8}'
    date = kwargs.get('date', '')
    
    
    raw_neuron_pt_tuple_files = sort_neuron_pt_tuple_file_list(raw_neuron_pt_tuple_folder)
    raw_neuron_pt_tuple = [np.load(f) for f in raw_neuron_pt_tuple_files]


    src_pt_tuple = neuron_pt_tuple_hat[root_idx, :, :7] # get the first 7 elements of the root time point

    for time_idx in range(neuron_pt_tuple_hat.shape[0]):
        src_pos_hat = neuron_pt_tuple_hat[time_idx, :, :7] # get the first 7 elements of the current time point
        tgt_pt_tuple = raw_neuron_pt_tuple[time_idx][:, :7] # get the first 7 elements of the current time point
        src_tgt_pair_dict = {
            'src': src_pt_tuple,
            'tgt': tgt_pt_tuple,
            'src_pos_hat': src_pos_hat
        }
        
        root_fname = os.path.splitext(os.path.basename(raw_neuron_pt_tuple_files[root_idx]))[0]
        curr_fname = os.path.splitext(os.path.basename(raw_neuron_pt_tuple_files[time_idx]))[0]

        if date != '':
            root_fname = re.sub(pattern, date, root_fname)
            curr_fname = re.sub(pattern, date, curr_fname)
        npz_file_path = os.path.join(npz_folder_path, f"{root_fname}_____{curr_fname}.npz")
        np.savez_compressed(npz_file_path, **src_tgt_pair_dict)


if __name__ == "__main__":
    neuron_pt_tuple_hat = np.load(r"I:\WJH\infer\manual\registration_annotation\20250730\w3_freelymoving\w3_manual\w3\neuron_pt_tuple.npy")
    raw_neuron_pt_tuple_folder = r"Y:\20250729_w3\self_training_data_straightened"
    npz_folder_path = r"I:\WJH\infer\manual\registration_annotation\20250730\w3_freelymoving\w3_manual\w3\npz_pair_1"
    neuron_pt_tuple2npz_pair(neuron_pt_tuple_hat, raw_neuron_pt_tuple_folder, npz_folder_path, root_idx=158, date='Dt202507')