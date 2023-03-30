import os
import sys
import json
import time
import itertools
import h5py

import numpy as np
import scipy.signal as sig

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

np.set_printoptions(linewidth=np.inf)


# Print
def print_off():
    sys.stdout = open(os.devnull, 'w')


def print_on():
    sys.stdout = sys.__stdout__


def print_update(sentence, i):
    """

    Args:
        sentence: sentence you want
        i: index in for loop

    Returns:

    """

    print(sentence, end='') if i == 0 else print('\r' + sentence, end='')


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    print("")


def print_info(args):
    print("")
    print(f"PID: {os.getpid()}\n")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print(f"Pytorch version: {torch.__version__}")
    print("")
    print_dict(args)


# Time
def convert_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    print(f"Total time: {h:02}:{m:02}:{s:02}")


def timeit(func):
    start = time.time()

    def decorator():
        _return = func()
        convert_time(time.time() - start)
        return _return

    return decorator


# Handling file an directory
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def write_json(path, json_data):
    with open(path, "w") as json_file:
        json.dump(json_data, json_file)


def read_json(path):
    with open(path, "r") as json_file:
        file = json.load(json_file)
    return file


def h5py_read(path):
    with h5py.File(path, 'r') as file:
        data = file['data'][()]
        return data


def make_save_path(path):
    if os.path.exists(path):
        return os.path.join(path, str(len(os.listdir(path))))
    else:
        return os.path.join(path, "0")


# Handling array
def order_change(array, order):
    array = list(array)
    tmp = array[order[0]]
    array[order[0]] = array[order[1]]
    array[order[1]] = tmp
    return array


def array_equal(A, B):
    return np.array_equal(np.round(A, 5), np.round(B, 5))


def convert_list(string):
    lst = string.split(",")
    assert len(lst) % 2 == 0, "Length of the list must be even number."
    it = iter(lst)
    return [list(map(int, itertools.islice(it, i))) for i in ([2] * (len(lst) // 2))]


def str2list_int(string):
    if string == 'all':
        return 'all'
    else:
        return list(map(int, string.split(",")))


def str2list(string):
    if string == 'all':
        return 'all'
    else:
        return string.split(",")


def str2dict(string):
    lst = string.split("_")
    return {key: str2list_int(value) for key, value in zip(lst[::2], lst[1::2])}


# Operation
def plv_signal(sig1, sig2):
    sig1_hill = sig.hilbert(sig1)
    sig2_hill = sig.hilbert(sig2)
    phase_1 = np.angle(sig1_hill)
    phase_2 = np.angle(sig2_hill)
    phase_diff = phase_1 - phase_2
    _plv = np.abs(np.mean([np.exp(complex(0, phase)) for phase in phase_diff]))
    return _plv


# Tensor operation  Note: version 만들기
def compatible_torch(func):
    def decorator(tensor, *args):
        _type = type(tensor)
        if _type == torch.Tensor:
            device = tensor.device.type
            if device == 'cpu':
                tensor = tensor.numpy()
            else:
                tensor = tensor.data.cpu().numpy()
        _return = func(tensor, *args)
        if _type == torch.Tensor:
            if device == 'cpu':
                _return = torch.Tensor(_return)
            else:
                _return = torch.Tensor(_return).cuda()
        return _return

    return decorator


def transpose_tensor(tensor, order):
    return np.transpose(tensor, order_change(np.arange(len(tensor.shape)), order))


def plv_tensor(tensor):
    """

    Parameters
    ----------
    tensor: [..., channels, times]

    Returns
    -------

    """
    tensor = np.angle(sig.hilbert(tensor))
    tensor = np.exp(tensor * 1j)
    _plv = np.abs(
        (tensor @ (np.transpose(tensor, order_change(np.arange(len(tensor.shape)), [-1, -2])) ** -1)) / np.size(tensor,
                                                                                                                -1))
    return _plv


@compatible_torch
def corr_tensor(tensor):
    """

    Parameters
    ----------
    tensor: [..., channels, times]

    Returns: channels * channels correlation coefficient matrix
    -------

    """
    mean = tensor.mean(axis=-1, keepdims=True)
    tensor2 = tensor - mean
    tensor3 = tensor2 @ np.transpose(tensor2, order_change(np.arange(len(tensor2.shape)), [-1, -2]))
    tensor4 = np.sqrt(np.expand_dims(np.diagonal(tensor3, axis1=-2, axis2=-1), axis=-1) @ np.expand_dims(
        np.diagonal(tensor3, axis1=-2, axis2=-1), axis=-2))
    corr = tensor3 / tensor4
    return corr


@compatible_torch
def normalize_adj_tensor(A):
    diag = np.power(A.sum(-2, keepdims=True), -0.5)
    diag[np.isinf(diag)] = 0.
    return transpose_tensor((A * diag), [-1, -2]) * diag


def segment_tensor(tensor, window_size, step):
    """

    Parameters
    ----------
    tensor: [..., chans, times]
    window_size
    step

    Returns: [shape[0], segment, ..., chans, times]
    -------

    """
    segment = []
    times = np.arange(tensor.shape[-1])
    start = times[::step]
    end = start + window_size
    for s, e in zip(start, end):
        if e > len(times):
            break
        segment.append(tensor[..., s:e])
    segment = transpose_tensor(np.array(segment), [0, 1])
    return segment


@compatible_torch
def apply_threshold(tensor, thr):
    '''

    Parameters
    ----------
    tensor: target tensor
    thr: the number of selection

    Returns
    -------

    '''
    shape = tensor.shape
    tensor = tensor.reshape(*list(tensor.shape)[:-2], -1)
    idx = np.flip(np.argsort(tensor, axis=1), axis=1)[..., :thr]
    idx = np.array([(n_row, v) for n_row, value in enumerate(idx) for v in value])
    tensor_selection = np.zeros_like(tensor)
    tensor_selection[idx[:, 0], idx[:, 1]] = 1
    tensor_selection = np.reshape(tensor_selection, shape)
    return tensor_selection


# Visualization
def compatible_torch_visualization(func):
    def decorator(tensor, *args):
        _type = type(tensor)
        if _type == torch.Tensor:
            device = tensor.device.type
            if device == 'cpu':
                tensor = tensor.numpy()
            else:
                tensor = tensor.data.cpu().numpy()
        _return = func(tensor, *args)
        return _return

    return decorator


# Note: Add text
@compatible_torch_visualization
def heatmap(matrix, name):
    sns.heatmap(matrix)
    plt.title(name)
    plt.show()


# Miscellaneous
def control_random(args):
    # Control randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu == "multi":
        torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    else:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False  # If you want set randomness, cudnn.benchmark = False
    cudnn.deterministic = True
    print(f"[Control randomness]\nseed: {args.seed}\n")


def net_debug(net, shape):
    input_data = torch.randn(shape)
    net(input_data)
    print("Finish network debugging.")


def output_shape(net, input_shape):
    X = torch.randn(input_shape[1:]).unsqueeze(dim=0)
    return net(X).shape


def uncuda(tensor):
    assert tensor.device.type == 'cuda', "Tensor device should be cuda."
    return tensor.data.cpu().numpy()


def cuda(tensor):
    return torch.FloatTensor(tensor).cuda()


def band_list(string):
    if string == 'all':
        return [[0, 4], [4, 7], [7, 13], [13, 30], [30, 42]]
    lst = string.split(",")
    assert len(lst) % 2 == 0, "Length of the list must be even number."
    it = iter(lst)
    return [list(map(int, itertools.islice(it, i))) for i in ([2] * (len(lst) // 2))]


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, in_tensor):
        return in_tensor.view((in_tensor.size()[0], -1))


def pprint(*args):
    out = [str(argument) + "\n" for argument in args]
    print(*out, "\n")