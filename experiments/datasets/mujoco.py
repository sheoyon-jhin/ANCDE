import collections as co
import numpy as np
import os
import pathlib
import sktime
import sktime.utils.load_data
import torch
import urllib.request
import zipfile
import time_dataset
from . import common

here = pathlib.Path(__file__).resolve().parent


def _process_data(input_seq,output_seq, missing_rate):
    PATH = os.path.dirname(os.path.abspath(__file__))
    
    torch.__version__
    X_times = np.load(PATH+"/mujoco.npy")
    X_times = torch.tensor(X_times)
    feature_lst = ['nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    final_indices = []
    for time in X_times:
        
        final_indices.append(len(time)-1)
    maxlen = max(final_indices)+1
    
    for i in range(len(X_times)):
    
        for _ in range(maxlen - len(X_times[i])):
            X_times[i].append([float('nan') for i in feature_lst])
    
    
    final_indices = torch.tensor(final_indices)

    X_reg = []
    y_reg = []
    for i in range(X_times.shape[0]):
        for j in range(X_times.shape[1]-input_seq-output_seq): 
            X_reg.append(X_times[i,j:j+input_seq,:].tolist())
            y_reg.append(X_times[i,j+input_seq:j+input_seq+output_seq,:].tolist())
    
    X_reg = torch.tensor(X_reg)
    y_reg = torch.tensor(y_reg)

    
    generator = torch.Generator().manual_seed(56789)
    for Xi in X_reg:
        removed_points = torch.randperm(X_reg.size(1), generator=generator)[:int(X_reg.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')
    final_indices_reg = np.repeat(input_seq-1,X_reg.shape[0])
    final_indices_reg = torch.tensor(final_indices_reg)
    
    
    times = torch.linspace(1, X_reg.size(1), X_reg.size(1))
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, input_channels) = common.preprocess_data_forecasting(times, X_reg, y_reg, final_indices_reg)
    

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, 1, input_channels)


def get_data(input_seq,output_seq, missing_rate, device, intensity, batch_size):
    
    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'Mujoco'
    dataset_name = "Mujoco"
    loc = base_loc / (dataset_name +'_'+ str(int(missing_rate * 100))+'_'+str(input_seq)+'_'+str(output_seq)) 
    # import pdb ; pdb.set_trace()
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        num_classes = int(tensors['num_classes'])
        input_channels = int(tensors['input_channels'])
    else:
        
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
        test_final_index, num_classes, input_channels) = _process_data(input_seq,output_seq, missing_rate)
        
        common.save_data(loc,
                         times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels))
    
    

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, device,
                                                                                num_workers=0, batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels
