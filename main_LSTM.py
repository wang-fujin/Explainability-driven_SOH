import torch
import torch.nn as nn
import os
import numpy as np
from utils.util import AverageMeter,save_to_txt,mkdir,create_logger,eval_metrix
from utils.LRP import get_LRP_weigth
from utils.LSTM_model import model,get_LRP
import matplotlib.pyplot as plt
import time
from utils.trainer_function import get_optimizer,get_scheduler,load_data,set_random_seed

import math

def final_test(test_loader,model,args):
    ground_true = []
    predict_label = []
    predict_label_lrp = []

    for data,label in test_loader:
        model.eval()

        data = data.to(args.device)
        pred = model(data)
        ground_true.append(label)
        predict_label.append(pred.cpu().detach().numpy())
        #################################################

        R2 = get_LRP(model,data)
        weight = get_LRP_weigth(R2[0],type=args.weight_type,scaler=args.weight_scaler)
        lrp_pred = model(data,weight)
        predict_label_lrp.append(lrp_pred.cpu().detach().numpy())
        #################################################

    return np.concatenate(ground_true), np.concatenate(predict_label), np.concatenate(predict_label_lrp)



def train(train_loader, valid_loader, test_loader,model,optimizer,lr_scheduler,args):
    if args.is_save_logging:
        mkdir(args.save_root)
        log_name = args.save_root + '/train info.log'
        log, consoleHander, fileHander= create_logger(filename=log_name)
        log.critical(args)
    else:
        log, consoleHander = create_logger()

    try:
        stop = 0
        min_test_loss = 10
        last_best_model = None
        criterion = nn.MSELoss()
        for e in range(1,args.n_epoch+1):
            model.train()
            pred_loss = AverageMeter()
            lrp_loss = AverageMeter()

            for data,label in train_loader:
                model.train()
                data = data.to(args.device)
                label = label.to(args.device)
                pred = model(data)
                pred_l = criterion(pred, label)

                #################################################
                if args.lrp_guided:
                    model.eval()
                    R2 = get_LRP(model,data)
                    weight = get_LRP_weigth(R2[0],type=args.weight_type,scaler=args.weight_scaler)
                    model.train()
                    lrp_pred = model(data,weight)
                    lrp_l = criterion(lrp_pred, label)
                    lrp_loss.update(lrp_l.item())
                    #################################################
                    loss = args.alpha * pred_l \
                           + args.beta * lrp_l
                else:
                    loss = pred_l

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred_loss.update(loss.item())
            if lr_scheduler:
                lr_scheduler.step()

            train_info = f'training......Epoch:[{e}/{args.n_epoch}], ' \
                         f'LRP guided={args.lrp_guided}, ' \
                         f'pred_loss:{100*pred_loss.avg:.4f}, ' \
                         f'lrp loss:{100*lrp_loss.avg:.4f}'
            log.info(train_info)

            ##################### test #######################
            if valid_loader is None:
                valid_loader = test_loader
            stop += 1
            #### valid
            true_label,pred_label,lrp_pred_label = final_test(model=model, test_loader=valid_loader, args=args)
            raw_matrix = eval_metrix(true_label=true_label,pred_label=pred_label)
            lrp_matrix = eval_metrix(true_label=true_label,pred_label=lrp_pred_label)
            valid_info = f"validing......valid_MAE:[{raw_matrix[0]:.4f}, {lrp_matrix[0]:.4f}]  lr:{optimizer.state_dict()['param_groups'][0]['lr']}"
            log.warning(valid_info)
            min_loss = min(raw_matrix[0],lrp_matrix[0])

            if min_test_loss > min_loss:
                min_test_loss = min_loss

                true_label,pred_label,lrp_pred_label = final_test(model=model, test_loader=test_loader, args=args)
                metrix = eval_metrix(true_label=true_label,pred_label=pred_label)
                lrp_matrix = eval_metrix(true_label=true_label,pred_label=lrp_pred_label)

                test_loss = metrix[2]*100
                test_info = f"testing......Epoch:[{e}/{args.n_epoch}],  test_loss(100x):{test_loss:.4f}," \
                            f"matrix:: MAE={metrix[0]:.6f},MAPE={metrix[1]:.6f},RMSE={metrix[3]:.6f}.   " \
                            f"lrp_matrix:: MAE={lrp_matrix[0]:.6f},MAPE={lrp_matrix[1]:.6f},RMSE={lrp_matrix[3]:.6f}."
                log.error(test_info)

                stop = 0
                #######plot test results#########
                if args.is_plot_test_results:
                    plt.plot(true_label, label='true')
                    plt.plot(pred_label, label='pred')
                    plt.plot(lrp_pred_label,label='lrp pred')
                    plt.title(f"Epoch:{e}, MAE:{metrix[0]:.4f}, {lrp_matrix[0]:.4f}")
                    plt.legend()
                    plt.show()
                ####### save model ########
                if args.is_save_best_model:
                    if last_best_model is not None and e<50:
                        os.remove(last_best_model)  # delete last best model

                    save_folder = args.save_root + '/pth'
                    mkdir(save_folder)
                    best_model = os.path.join(save_folder, f'Epoch{e}.pth')
                    torch.save(model.state_dict(), best_model)
                    last_best_model = best_model
                #########save test results (test info) to txt #####
                if args.is_save_to_txt:
                    txt_path = args.save_root + '/test_info.txt'
                    time_now = time.strftime("%Y-%m-%d", time.localtime())
                    if e == 1:
                        save_to_txt(txt_path, ' ')
                        #save_to_txt(txt_path,f'########## experiment {args.experiment_time} ##########')
                        for k,v in vars(args).items():
                            save_to_txt(txt_path,f'{k}:{v}')
                    info = time_now + f' epoch = {e}, test_loss(100x):{test_loss:.6f}， ' \
                                      f'matrix:: MAE={metrix[0]:.6f},MAPE={metrix[1]:.6f},RMSE={metrix[3]:.6f}. \n' \
                                      f'matrix_lrp:: MAE={lrp_matrix[0]:.6f},MAPE={lrp_matrix[1]:.6f},RMSE={lrp_matrix[3]:.6f}'
                    save_to_txt(txt_path,info)
                #########save test results (predict value) to np ######
                if args.is_save_test_results:
                    np.save(args.save_root+'/pred_label',pred_label)
                    np.save(args.save_root+'/true_label',true_label)
                    np.save(args.save_root+'/lrp_pred_label',lrp_pred_label)
            if args.early_stop > 0 and stop > args.early_stop:
                print(' Early Stop !')
                if args.is_save_logging:
                    log.removeHandler(consoleHander)
                    log.removeHandler(fileHander)
                else:
                    log.removeHandler(consoleHander)
                break

    except:
        txt_path = args.save_root + '/test_info.txt'
        save_to_txt(txt_path, 'Error !')
    if args.is_save_logging:
        log.removeHandler(consoleHander)
        log.removeHandler(fileHander)
    else:
        log.removeHandler(consoleHander)


def main(args):
    set_random_seed(args.seed)
    _, train_loader, valid_loader, test_loader = load_data(args)
    m = model().to(args.device)

    optimizer = get_optimizer(m, args)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(train_loader, valid_loader, test_loader, m,optimizer, scheduler, args)
    #torch.cuda.empty_cache()
    del train_loader
    del valid_loader
    del test_loader


def run_on_MIT_for_weight_select():
    from utils.config import get_args
    args = get_args()
    for i in range(5):  # 5次实验
        for test_id in [1, 2, 3, 4, 5]:
            for data_set in ['MIT2_1', 'MIT2_2']:
                for w in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
                    if w == 0:
                        setattr(args, 'lrp_guided', False)
                        setattr(args, 'save_root',
                                f'experiments2/weight select/MIT2/LRP guide False/{data_set}/test battery {test_id}/experiment {i}')
                    else:
                        setattr(args, 'lrp_guided', True)
                        setattr(args, 'save_root',
                                f'experiments2/weight select/MIT2/LRP guide True_{w}/{data_set}/test battery {test_id}/experiment {i}')
                    try:
                        setattr(args, 'source_dir', f'data/MIT/{data_set}')
                        setattr(args, 'is_save_best_model', False)
                        setattr(args, 'weight_scaler', w)
                        setattr(args, 'test_id', test_id)
                        setattr(args, 'is_DA',False)
                        main(args)
                    except:
                        continue


def run_on_all_MIT_data():
    from utils.config import get_args
    args = get_args()
    for i in range(5):
        for test_id in [1,2,3,4,5]:
            for lrp in [True,False]:
                for data in ['MIT2_2','MIT2_3','MIT2_4','MIT2_5']:
                    try:
                        setattr(args, 'early_stop', 0)
                        setattr(args, 'is_save_best_model', False)
                        setattr(args, 'source_dir', f'data/MIT/{data}')
                        setattr(args, 'lrp_guided', lrp)
                        setattr(args, 'test_id', test_id)
                        setattr(args, 'save_root',
                                f'experiments2/MIT2-LSTM/LRP guide {lrp}/{data}/test battery {test_id}/experiment {i}')
                        main(args)
                    except:
                        continue


def run_on_all_BIT_data():
    from utils.config import get_args
    args = get_args()
    for i in range(5): # 5 experiments
        for test_id in [1, 2, 3, 4, 5, 6, 7, 8]:
            for lrp in [True,False]:
                for data in ['BIT-1', 'BIT-2']:
                    try:
                        setattr(args, 'source_dir', f'data/BIT/{data}')
                        setattr(args, 'is_save_best_model', False)
                        setattr(args, 'lrp_guided', lrp)
                        setattr(args, 'test_id', test_id)
                        setattr(args, 'save_root',
                                f'experiments2/BIT-LSTM-2/LRP guide {lrp}/{data}/test battery {test_id}/experiment {i}')
                        main(args)
                    except:
                        continue

def run_on_all_CALCE_data():
    from utils.config import get_args
    args = get_args()
    for i in range(5):  # 5 experiments
        for test_id in [1, 2, 3, 4, 5, 6]:
            for lrp in [True, False]:
                for data in ['CS2', 'CX2']:
                    try:
                        setattr(args, 'is_save_best_model', False)
                        setattr(args, 'source_dir', f'data/CALCE/{data}')
                        setattr(args, 'lrp_guided', lrp)
                        setattr(args, 'test_id', test_id)
                        setattr(args, 'save_root',
                                f'experiments2/CALCE-LSTM/LRP guide {lrp}/{data}/test battery {test_id}/experiment {i}')
                        main(args)
                    except:
                        continue

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    run_on_all_BIT_data()




