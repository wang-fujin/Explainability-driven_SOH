import torch
import numpy as np
from utils.util import AverageMeter,eval_metrix


def set_random_seed(seed=2022):
    import random
    random.seed(seed)
    np.random.seed(seed)

def get_optimizer(model,args):
    initial_lr = args.lr #if not args.lr_scheduler else 1.0
    params = model.parameters()
    optimizer = torch.optim.Adam(params,lr=initial_lr,weight_decay=args.weight_decay)
    return optimizer

# def get_scheduler(optimizer,args):
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
#         -args.lr_decay))
#     return scheduler


def get_scheduler(optimizer,args):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        args.lr_step,
        args.lr_gamma,
    )
    return scheduler

class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

def test(model,target_test_loader,args):
    model.eval()
    test_loss = AverageMeter()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        ground_true = []
        predict_label = []
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss = criterion(output,target)
            test_loss.update(loss.item())
            ground_true.append(target.cpu().detach().numpy())
            predict_label.append(output.cpu().detach().numpy())
    true_label = np.concatenate(ground_true)
    pred_label = np.concatenate(predict_label)
    return test_loss.avg, true_label, pred_label, eval_metrix(true_label,pred_label)


def load_data(args):
    if 'BIT' in args.source_dir:
        from dataloader.BIT_loader import load_single_domain_data,load_multi_domain_data
    elif 'MIT' in args.source_dir:
        from dataloader.MIT_loader import load_single_domain_data,load_multi_domain_data
    elif 'CALCE' in args.source_dir:
        from dataloader.CALCE_loader import load_single_domain_data

    if args.is_DA:
        source_loader, target_train_loader, target_valid_loader, target_test_loader = load_multi_domain_data(args)
        return (source_loader, target_train_loader, target_valid_loader, target_test_loader)
    else:
        train_loader, valid_loader, test_loader = load_single_domain_data(args)
        return (None,train_loader, valid_loader, test_loader)

if __name__ == '__main__':
    pass