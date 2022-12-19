import argparse

__all__ = ['get_args']

def get_args():
    parser = argparse.ArgumentParser(description='Explainable prediction for battery capacity !')
    parser.add_argument('--input_channel',default=3)

    # data
    parser.add_argument('--source_dir',type=str, default='data/MIT/MIT2_2')
    #parser.add_argument('--target_dir',type=str, default='data/MIT/MIT2_5')
    parser.add_argument('--test_id',type=int, default=2)#,choices=[1,2,3,4,5])

    parser.add_argument('--normalize_type',default='minmax',choices=['minmax','standerd'])
    parser.add_argument('--batch_size',type=int, default=128)
    parser.add_argument('--seed',default=2022)

    # optimizer
    parser.add_argument('--lr',type=float, default=2e-3) #1e-3
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler
    # parser.add_argument('--lr_gamma', type=float, default=0.0003)
    # parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_step', type=list,default=[30,50])
    parser.add_argument('--lr_gamma',type=float,default=0.5)
    parser.add_argument('--lr_scheduler', type=bool, default=True)

    parser.add_argument('--device',default='cuda')
    parser.add_argument('--n_epoch',default=100)
    parser.add_argument('--early_stop', default=20)

    parser.add_argument('--alpha', default=1)
    parser.add_argument('--beta',default=0.9)
    #parser.add_argument('--lambda_',default=0.005) # MMD loss

    # DA
    parser.add_argument('--is_DA',type=bool,default=False)

    # LRP parameters
    parser.add_argument('--lrp_guided',type=bool,default=True)
    parser.add_argument('--weight_type',type=str,default='minmax',choices=['minmax','uniform'])
    parser.add_argument('--weight_scaler',type=float,default=0.5)

    # reulsts
    parser.add_argument('--is_plot_test_results',default=False)
    parser.add_argument('--is_save_logging',default=True)
    parser.add_argument('--is_save_best_model',default=True)
    parser.add_argument('--is_save_to_txt',default=True)
    parser.add_argument('--is_save_test_results',default=True)
    parser.add_argument('--experiment_time',type=int,default=1)

    parser.add_argument('--save_root',default='experiment/just_test')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    args = vars(args)
    for i in args.items():
        print(i)
