import os
import pathlib
from typing_extensions import final
import sklearn.model_selection
import sys
import torch
import pickle
here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq


def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 8
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def split_data(tensor, stratify,socar=False):
    # 0.7/0.15/0.15 train/val/test split
    if socar :
        print("setting : socar - True")
        # 150 : 499* 0.3
        train_tensor = None
        val_tensor, test_tensor = sklearn.model_selection.train_test_split(tensor,
                                                                        train_size=0.5,
                                                                        random_state=1,
                                                                        shuffle=True,
                                                                        stratify=stratify)
    else:
        print("setting : socar - False")
        (train_tensor, testval_tensor,
        train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                    train_size=0.7,
                                                                                    random_state=0,
                                                                                    shuffle=True,
                                                                                    stratify=stratify)

        val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                        train_size=0.5,
                                                                        random_state=1,
                                                                        shuffle=True,
                                                                        stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y) 
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def preprocess_data(times, X, y, final_index, append_times, append_intensity):
    
    X = normalise_data(X.cpu(), y.cpu()).cuda() 
    # X = normalise_data(X, y).cuda() 
    X = X.cuda()
    print(f"in X.shape {X.shape}") 
    augmented_X = []
    if append_times:
        augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    if append_intensity:
        intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
        intensity = intensity.to(X.dtype).cumsum(dim=1)
        augmented_X.append(intensity)
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)

    
    
    
    train_X, val_X, test_X = split_data(X, y)
    train_y, val_y, test_y = split_data(y, y)
    
    train_final_index, val_final_index, test_final_index = split_data(final_index, y)
    import pdb ; pdb.set_trace()
    
     
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), train_X)
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), val_X)
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times.cuda(), test_X)
    
    in_channels = X.size(-1)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels)


def preprocess_data_forecasting(times, X, y, final_index, append_times, append_intensity):
    
    # import pdb ; pdb.set_trace()
    X = X.cuda()
    print(f"in X.shape {X.shape}") 
    
    train_X, train_y = X[:2564], y[:2564]
    val_X, val_y = X[2564:3113],y[2564:3113]
    test_X, test_y = X[3113:], y[3113:]
    
    
    train_final_index, val_final_index, test_final_index = final_index[:2564],final_index[2564:3113],final_index[3113:]

    # import pdb ; pdb .set_trace()
    # PATH = '/home/bigdyl/socar/NeuralCDE/experiments/datasets/processed_data/Stock_raw/70/'
    # torch.save(train_X, PATH + 'train_X.pt')
    # torch.save(val_X, PATH + 'val_X.pt')
    # torch.save(test_X, PATH + 'test_X.pt')
    # torch.save(train_y, PATH + 'train_y.pt')
    # torch.save(val_y,  PATH + 'val_y.pt')
    # torch.save(test_y,  PATH + 'test_y.pt')
    # torch.save(train_final_index,  PATH + 'train_final_index.pt')
    # torch.save(val_final_index,  PATH + 'val_final_index.pt')
    # torch.save(test_final_index,  PATH + 'test_final_index.pt')
    # # exit()
    # import pdb; pdb.set_trace() # temporary stop to create dataset
    
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_X.cuda())
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_X.cuda())
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_X.cuda())
    
    in_channels = X.size(-1)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels)


def preprocess_data_socar(times, X, y, final_index, append_times, append_intensity):

    
    X = normalise_data(X, y) # torch.Size([2858, 182, 3])
    print(f"in X.shape {X.shape}")
    augmented_X = []
    if append_times:
        augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    if append_intensity: 
        
        intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
        intensity = intensity.to(X.dtype).cumsum(dim=1)
        augmented_X.append(intensity)
    augmented_X.append(X)
    if len(augmented_X) == 1:
        X = augmented_X[0]
    else:
        X = torch.cat(augmented_X, dim=2)
    
    i = 160
    train_X= X[:i,:,:]
    train_y = y[:i]
    train_final_index= final_index[:i]
    
    X2 = X[i:,:,:]
    y2 = y[i:]
    final_index2 = final_index[i:]
    _,val_X, test_X = split_data(X2, y2,socar=True)
    _,val_y, test_y = split_data(y2, y2,socar=True)
    _, val_final_index, test_final_index = split_data(final_index2, y2,socar=True)
    print(f"Train Class Ratio at {i}: 0 : {(i-torch.sum(train_y))/i} 1 : {torch.sum(train_y)/i}")
    print(f"Val Class Ratio   at {i}: 0 : {(val_y.shape[0]-torch.sum(val_y))/val_y.shape[0]} 1 : {torch.sum(val_y)/val_y.shape[0]}")
    print(f"Test Class Ratio  at {i}: 0 : {(test_y.shape[0]-torch.sum(test_y))/test_y.shape[0]} 1 : {torch.sum(test_y)/test_y.shape[0]}")
    
    print("START Extrapolation")
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_X)
    print(" >> finish Interpolate_[train]")
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_X)
    print(" >> finish Interpolate_[validation]")
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_X)
    print(" >> finish Interpolate_[test]")
        
    
    in_channels = X.size(-1)
    print(f"in_channels {in_channels}")
    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels)



def wrap_data(times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
              test_final_index, device, batch_size, num_workers=4):
    times = times.to(device)
    train_coeffs = tuple(coeff.to(device) for coeff in train_coeffs)
    val_coeffs = tuple(coeff.to(device) for coeff in val_coeffs)
    test_coeffs = tuple(coeff.to(device) for coeff in test_coeffs)
    train_y = train_y.to(device)
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    train_final_index = train_final_index.to(device)
    val_final_index = val_final_index.to(device)
    test_final_index = test_final_index.to(device)
    
    train_dataset = torch.utils.data.TensorDataset(*train_coeffs, train_y, train_final_index)
    val_dataset = torch.utils.data.TensorDataset(*val_coeffs, val_y, val_final_index)
    test_dataset = torch.utils.data.TensorDataset(*test_coeffs, test_y, test_final_index)

    train_dataloader = dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return times, train_dataloader, val_dataloader, test_dataloader


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors
