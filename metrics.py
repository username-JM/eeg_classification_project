import numpy as np
from sklearn.metrics import accuracy_score as accuracy


# If you want to use custom metrics, please add functions.

def cal_log(log_dict, **kwargs):
    for key, value in log_dict.items():
        value.append(globals()[key.split("_")[-1]](kwargs))
    return log_dict


def loss(kwargs):
    return float(kwargs['loss'].data.cpu().numpy())


def acc(kwargs):
    return accuracy(np.argmax(kwargs['outputs'].data.cpu().numpy(), axis=1), kwargs['labels'].data.cpu().numpy())
