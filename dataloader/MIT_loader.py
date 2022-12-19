import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from utils.config import get_args
import os
from utils.util import Scaler
from sklearn.model_selection import train_test_split


def load_single_domain_data(args):
    X_files = os.listdir(os.path.join(args.source_dir, 'X'))
    Y_files = os.listdir(os.path.join(args.source_dir, 'Y'))

    ############################
    ####### load data
    ############################
    count = 0
    X_list = []
    Y_list = []
    for x in X_files:
        count += 1
        for y in Y_files:
            if x.split('_')[2] == y.split('_')[2]:
                if count == args.test_id:
                    test_X = np.load(os.path.join(args.source_dir, 'X', x))[10:-20]
                    test_Y = np.load(os.path.join(args.source_dir, 'Y', y))[10:-20]
                    test_X = Scaler(test_X).minmax()
                    test_Y = Scaler(test_Y).max_normalize()
                    print(f'target test battery: {args.test_id} -> {x}')
                    continue

                battery_i_data = np.load(os.path.join(args.source_dir, 'X', x))[10:-20]
                battery_i_capacity = np.load(os.path.join(args.source_dir, 'Y', y))[10:-20]
                data_x = Scaler(battery_i_data).minmax()
                data_y = Scaler(battery_i_capacity).max_normalize()
                X_list.append(data_x)
                Y_list.append(data_y)
                break
    train_X = np.concatenate(X_list, axis=0)
    train_Y = np.concatenate(Y_list, axis=0)
    print('=' * 50)
    print('MIT data:')
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

    source_X_files = os.listdir(os.path.join(args.source_dir,'X'))
    source_Y_files = os.listdir(os.path.join(args.source_dir, 'Y'))

    target_X_files = os.listdir(os.path.join(args.target_dir,'X'))
    target_Y_files = os.listdir(os.path.join(args.target_dir,'Y'))


    ############################
    ####### load source data
    ############################
    Xs_list = []
    Ys_list = []

    for x in source_X_files:
        for y in source_Y_files:
            if x.split('_')[2] == y.split('_')[2]:
                battery_i_data = np.load(os.path.join(args.source_dir,'X',x))[10:-20]
                battery_i_capacity = np.load(os.path.join(args.source_dir,'Y',y))[10:-20]
                data_x = Scaler(battery_i_data).minmax()
                data_y = Scaler(battery_i_capacity).max_normalize()
                Xs_list.append(data_x)
                Ys_list.append(data_y)
                break

    source_X = np.concatenate(Xs_list,axis=0)
    source_Y = np.concatenate(Ys_list,axis=0)
    print('='*50)
    print('MIT data:')
    print(f'source ({args.source_dir}): {source_X.shape}, {source_Y.shape}')

    ############################
    ####### load target data
    ############################
    count = 0
    Xt_list = []
    Yt_list = []
    for x in target_X_files:
        count += 1
        for y in target_Y_files:
            if x.split('_')[2] == y.split('_')[2]:
                if count == args.test_id:
                    target_test_X = np.load(os.path.join(args.target_dir,'X',x))[10:-20]
                    target_test_X = Scaler(target_test_X).minmax()
                    target_test_Y = np.load(os.path.join(args.target_dir, 'Y', y))[10:-20]
                    target_test_Y = Scaler(target_test_Y).max_normalize()

                    print(f'target test battery ({args.target_dir}): {x}')
                    continue

                battery_i_data = np.load(os.path.join(args.target_dir,'X',x))[10:-20]
                battery_i_capacity = np.load(os.path.join(args.target_dir,'Y',y))[10:-20]
                data_x = Scaler(battery_i_data).minmax()
                data_y = Scaler(battery_i_capacity).max_normalize()
                Xt_list.append(data_x)
                Yt_list.append(data_y)
                break
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
        setattr(args, 'source_dir', '../data/MIT/MIT2_5')
        setattr(args, 'test_id',i)
        data_set = args.source_dir.split('/')[-1]
        load_single_domain_data(args)

    #load_multi_domain_data(args)