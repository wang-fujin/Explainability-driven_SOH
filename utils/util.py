import torch
import torch.nn as nn
import copy
import logging
import numpy as np
import os
import colorlog
from sklearn import metrics
# --------------------------------------------------------------
# Clone a layer and pass its parameters through the function g
# --------------------------------------------------------------
def newlayer(layer,g):
    layer = copy.deepcopy(layer)
    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer

# --------------------------------------------------------------
# convert VGG dense layers to convolutional layers
# --------------------------------------------------------------
def toconv(layers):
    newlayers = []
    for i,layer in enumerate(layers):
        if isinstance(layer,nn.Linear):
            newlayer = None
            if i == 0:
                m,n = 512,layer.weight.shape[0]
                newlayer = nn.Conv1d(m,n,4)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,4))
            else:
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv1d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]

    return newlayers


def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))
    return [MAE,MAPE,MSE,RMSE]



def save_to_txt(save_name,string):
    f = open(save_name,mode='a')
    f.write(string)
    f.write('\n')
    f.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Scaler():
    def __init__(self,data1,data2=None):  # data.shape (N,L,C)  或者 (N,C)
        if data2 is None:
            self.data = data1
        else:
            self.train_num = data1.shape[0]
            self.data = np.concatenate((data1, data2), axis=0)
        self.data2 = data2
        if self.data.ndim == 3:
            self.mean = self.data.mean(axis=(0,1)).reshape(1,1,-1)
            self.var = self.data.var(axis=(0,1)).reshape(1,1,-1)
            self.max = self.data.max(axis=(0,1)).reshape(1,1,-1)
            self.min = self.data.min(axis=(0,1)).reshape(1,1,-1)
        elif self.data.ndim ==2:
            self.mean = self.data.mean(axis=0).reshape(1, -1)
            self.var = self.data.var(axis=0).reshape(1, -1)
            self.max = self.data.max(axis=0).reshape(1, -1)
            self.min = self.data.min(axis=0).reshape(1, -1)
        elif self.data.ndim == 1: #标签数据
            self.mean = self.data.mean()
            self.var = self.data.var()
            self.max = self.data.max()
            self.min = self.data.min()
        else:
            raise ValueError('data dim error!')

    def standerd(self):
        X = (self.data - self.mean) / self.var
        if self.data2 is None:
            return X
        else:
            train = X[:self.train_num]
            test = X[self.train_num:]
            return train, test

    def minmax(self,is_zero_one=True):
        if is_zero_one:
            X = (self.data - self.min) / (self.max - self.min)
        else:
            X = 2*(self.data - self.min) / (self.max - self.min)-1
        if self.data2 is None:
            return X
        else:
            train = X[:self.train_num]
            test = X[self.train_num:]
            return train, test

    def max_normalize(self):
        X = self.data / self.max
        if self.data2 is None:
            return X
        else:
            train = X[:self.train_num]
            test = X[self.train_num:]
            return train, test



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_logger(filename=None,con_level='debug',file_level='debug'):
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'ctirical': logging.CRITICAL
    }

    logger1 = logging.getLogger('my_log')
    logger1.setLevel(logging.DEBUG)
    #处理器Hander
    consoleHander = logging.StreamHandler()
    consoleHander.setLevel(level_dict.get(con_level))
    #不指定输出级别则采用logger的级别,都设置了则取最低级
    if filename is not None:
        fileHander = logging.FileHandler(filename)
        fileHander.setLevel(level_dict.get(file_level))
    #formatter格式
    formatter1 = logging.Formatter("[%(asctime)s.%(msecs)d] -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s")
    formatter2 = colorlog.ColoredFormatter(fmt='%(log_color)s[%(asctime)s.%(msecs)03d] -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
                                           log_colors=log_colors_config)
    #给处理器设置格式
    consoleHander.setFormatter(formatter2)
    if filename is not None:
        fileHander.setFormatter(formatter1)
    #记录器设置处理器
    logger1.addHandler(consoleHander)
    if filename is not None:
        logger1.addHandler(fileHander)
        return logger1, consoleHander, fileHander
    else:
        return logger1, consoleHander