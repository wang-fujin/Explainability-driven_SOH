import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from utils.config import get_args
import os
from utils.util import Scaler
from sklearn.model_selection import train_test_split
import h5py

def get_h5_data(path):
    with h5py.File(path,'r') as f:
        x = np.array(f.get('charge_data'),dtype=np.float32)
        capacity = np.array(f.get('discharge_capacity'),dtype=np.float32)

    return x[:,:,:3],capacity[:]


def load_single_domain_data(args):
    print('='*50)
    print('BIT data (single domain)')

    ############################
    ####### load data
    ############################
    source_files = os.listdir(args.source_dir)
    count = 0
    X_list = []
    Y_list = []
    for file in source_files:
        count += 1

        if count == args.test_id:
            test_X, test_Y = get_h5_data(os.path.join(args.source_dir,file))

            test_X = Scaler(test_X).max_normalize()[1:]
            test_Y = Scaler(test_Y).max_normalize()[1:]
            print(f'target test battery (id={args.test_id}): {file}')
            continue

        x, y =get_h5_data(os.path.join(args.source_dir,file))
        data_x = Scaler(x).max_normalize()[1:]
        data_y = Scaler(y).max_normalize()[1:]
        X_list.append(data_x)
        Y_list.append(data_y)

    train_X = np.concatenate(X_list, axis=0)
    train_Y = np.concatenate(Y_list, axis=0)

    print(f'train(valid): {train_X.shape}, {train_Y.shape}')
    print(f'test:  {test_X.shape}, {test_Y.shape}')
    print('-' * 50)
    train_x = torch.from_numpy(np.transpose(train_X, (0, 2, 1)))
    train_y = torch.from_numpy(train_Y).view(-1, 1)
    test_x = torch.from_numpy(np.transpose(test_X, (0, 2, 1)))
    test_y = torch.from_numpy(test_Y).view(-1, 1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=args.seed)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def load_multi_domain_data(args):
    print('=' * 50)
    print('BIT data')

    ############################
    ####### load source data
    ############################
    source_files = os.listdir(args.source_dir)
    target_files = os.listdir(args.target_dir)

    Xs_list = []
    Ys_list = []

    for file in source_files:
        x,y = get_h5_data(os.path.join(args.source_dir,file))
        data_x = Scaler(x).max_normalize()
        data_y = Scaler(y).max_normalize()
        Xs_list.append(data_x)
        Ys_list.append(data_y)

    source_X = np.concatenate(Xs_list,axis=0)
    source_Y = np.concatenate(Ys_list,axis=0)

    print(f'source: {source_X.shape}, {source_Y.shape}')

    ############################
    ####### load target data
    ############################
    count = 0
    Xt_list = []
    Yt_list = []
    for file in target_files:
        count += 1
        if count == args.test_id:
            target_test_X,target_test_Y = get_h5_data(os.path.join(args.target_dir,file))

            target_test_X = Scaler(target_test_X).max_normalize()
            target_test_Y = Scaler(target_test_Y).max_normalize()
            print(f'target test battery: {file}')
            continue

        x, y = get_h5_data(os.path.join(args.target_dir, file))
        data_x = Scaler(x).max_normalize()
        data_y = Scaler(y).max_normalize()
        Xt_list.append(data_x)
        Yt_list.append(data_y)

    target_train_X = np.concatenate(Xt_list,axis=0)
    target_train_Y = np.concatenate(Yt_list,axis=0)
    print(f'target train: {target_train_X.shape}, {target_train_Y.shape}')
    print(f'target test:  {target_test_X.shape}, {target_test_Y.shape}')

    target_train_x = torch.from_numpy(np.transpose(target_train_X, (0, 2, 1)))
    target_train_y = torch.from_numpy(target_train_Y).view(-1, 1)
    target_test_x = torch.from_numpy(np.transpose(target_test_X, (0, 2, 1)))
    target_test_y = torch.from_numpy(target_test_Y).view(-1, 1)
    source_x = torch.from_numpy(np.transpose(source_X, (0, 2, 1)))
    source_y = torch.from_numpy(source_Y).view(-1, 1)

    ##########################################

    print('-'*50)

    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=args.batch_size, shuffle=True,drop_last=False)
    target_train_loader = DataLoader(TensorDataset(target_train_x, target_train_y), batch_size=args.batch_size,shuffle=True,drop_last=False)
    target_valid_loader = DataLoader(TensorDataset(target_train_x, target_train_y), batch_size=args.batch_size,shuffle=False,drop_last=False)
    target_test_loader = DataLoader(TensorDataset(target_test_x, target_test_y), batch_size=args.batch_size,shuffle=False)
    return source_loader, target_train_loader, target_valid_loader,target_test_loader


if __name__ == '__main__':
    args = get_args()
    for i in range(1,6):
        setattr(args, 'source_dir', '../data/BIT/BIT-2')
        setattr(args, 'test_id',i)
        data_set = args.source_dir.split('/')[-1]
        load_single_domain_data(args)


