import collections as co
import numpy as np
import os
import pathlib
import sktime
import sktime.utils.load_data
import torch
import urllib.request
import zipfile

from . import common

here = pathlib.Path(__file__).resolve().parent


def download():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'UEA'
    loc = base_loc / 'Multivariate2018_ts.zip'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip',
                               str(loc))

    with zipfile.ZipFile(loc, 'r') as f:
        f.extractall(str(base_loc))


# Is this actually necessary?
def _pad(channel, maxlen):
    # X중 하나의 데이터 들어옴 (Series) - (116,)
    channel = torch.tensor(channel) # Series를 tensor로 바꿈
    out = torch.full((maxlen,), channel[-1]) # tensor의 마지막 원소를 (maxlen,) 크기로 새로운 텐서 out을 만듦
    out[:channel.size(0)] = channel # 텐서 out의 원래 범위 만큼을 원래 값으로 채움
    return out # 리턴

valid_dataset_names = {'ArticularyWordRecognition',
                       'FaceDetection',
                       'NATOPS',
                       'AtrialFibrillation',
                       'FingerMovements',
                       'PEMS-SF',
                       'BasicMotions',
                       'HandMovementDirection',
                       'PenDigits',
                       'CharacterTrajectories',
                       'Handwriting',
                       'PhonemeSpectra',
                       'Cricket',
                       'Heartbeat',
                       'RacketSports',
                       'DuckDuckGeese',
                       'InsectWingbeat',
                       'SelfRegulationSCP1',
                       'EigenWorms',
                       'JapaneseVowels',
                       'SelfRegulationSCP2',
                       'Epilepsy',
                       'Libras',
                       'SpokenArabicDigits',
                       'ERing',
                       'LSST',
                       'StandWalkJump',
                       'EthanolConcentration',
                       'MotorImagery',
                       'UWaveGestureLibrary'}

def _process_data(dataset_name, missing_rate, intensity):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_filename = here / 'data' / 'UEA' / 'Multivariate_ts' / dataset_name / dataset_name
    train_X, train_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    test_X, test_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TEST.ts')
    import pdb;pdb.set_trace()

    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    # TODO
    X = np.concatenate((train_X, test_X), axis=0) # 2858,3  -> 
    y = np.concatenate((train_y, test_y), axis=0)

    

    lengths = torch.tensor([len(Xi[0]) for Xi in X]) # 각각의 길이 받음
    final_index = lengths - 1 # index는 0부터시작하니까 length의 각 값에서 1씩 뺌
    maxlen = lengths.max() # lengths 중에서 가장 긴 length = maxlen

    # X is now a numpy array of shape (batch, channel)
    # Each channel is a pandas.core.series.Series object of length corresponding to the length of the time series
    X = torch.stack([torch.stack([_pad(channel, maxlen) for channel in batch], dim=0) for batch in X], dim=0) # torch.Size([2858, 3, 182])
    
    # X is now a tensor of shape (batch, channel, length)
    X = X.transpose(-1, -2) #torch.Size([2858, 182, 3])
    # X is now a tensor of shape (batch, length, channel)
    times = torch.linspace(0, X.size(1) - 1, X.size(1)) # 0 ~ 181

    # missing_rate != 0 이면 missing_rate 만큼 nan 값으로 대체
    generator = torch.Generator().manual_seed(56789)
    for Xi in X:
        removed_points = torch.randperm(X.size(1), generator=generator)[:int(X.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')

    # Now fix the labels to be integers from 0 upwards
    targets = co.OrderedDict()
    counter = 0
    for yi in y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    
    # counter : OrderedDict([('1', 0), ('2', 1), ... ,('19', 18), ('20', 19)])

    y = torch.tensor([targets[yi] for yi in y]) # target을 targets의 value(counter 값)으로 바꿈

    # X : torch.Size([2858, 182, 3])
    # y : torch.Size([2858])
    # final_index : torch.Size([2858]) (length - 1)
    
    X=X.cuda()
    y=y.cuda()
    times = times.cuda()
    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, input_channels) = common.preprocess_data(times, X, y, final_index, append_times=True,
                                                                append_intensity=intensity)

    num_classes = counter

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, num_classes, input_channels)


def get_data(dataset_name, missing_rate, device, intensity, batch_size):
    # We begin by loading both the train and test data and using our own train/val/test split.
    # The reason for this is that (a) by default there is no val split and (b) the sizes of the train/test splits are
    # really janky by default. (e.g. LSST has 2459 training samples and 2466 test samples.)
    
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'UEA'
    loc = base_loc / (dataset_name + str(int(missing_rate * 100)) + ('_intensity' if intensity else ''))
    
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
        
        download()
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index, num_classes, input_channels) = _process_data(dataset_name, missing_rate, intensity)
        common.save_data(loc,
                         times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels))
    # train_coeffs : 2000,181,4 
    # val,test_coeffs : 429,181,4 
    
    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, device,
                                                                                num_workers=0, batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels
