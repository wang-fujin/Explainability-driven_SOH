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
                    test_X = np.load(os.path.join(args.source_dir, 'X', x)).astype(np.float32)
                    test_Y = np.load(os.path.join(args.source_dir, 'Y', y)).astype(np.float32)
                    test_X = Scaler(test_X).minmax()
                    test_Y = Scaler(test_Y).max_normalize()
                    print(f'target test battery (id={args.test_id}): {x}')
                    continue

                battery_i_data = np.load(os.path.join(args.source_dir, 'X', x))
                battery_i_capacity = np.load(os.path.join(args.source_dir, 'Y', y))
                data_x = Scaler(battery_i_data).minmax()
                data_y = Scaler(battery_i_capacity).max_normalize()
                X_list.append(data_x)
                Y_list.append(data_y)
                break
    train_X = np.concatenate(X_list, axis=0).astype(np.float32)
    train_Y = np.concatenate(Y_list, axis=0).astype(np.float32)
    print('=' * 50)
    print('CALCE data:')
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




if __name__ == '__main__':
    args = get_args()
    for i in range(1, 6):
        setattr(args, 'source_dir', '../data/CALCE/CX2')
        setattr(args, 'test_id', i)
        data_set = args.source_dir.split('/')[-1]
        load_single_domain_data(args)


