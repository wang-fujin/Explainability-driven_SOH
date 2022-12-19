import torch
import utils.util as util
from utils.model import model
import numpy as np
from utils.config import get_args
from dataloader.MIT_loader import load_single_domain_data as load_data
import matplotlib.pyplot as plt


# def heat_map(A,R):
#     b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3)) # the smaller the b, the deeper the color
#
#     from matplotlib.colors import ListedColormap
#     my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
#     my_cmap[:, 0:3] *= 0.85
#     my_cmap = ListedColormap(my_cmap)
#     #fig = plt.figure(figsize=(len(R) * 4 / 100, 2), dpi=100)
#     fig = plt.figure(figsize=(128 * 4 / 100, 2),dpi=100)
#     ax1 = fig.add_subplot(111)
#     ax1.set_yticks([])
#     ax1.imshow(R.reshape(1, -1), cmap=my_cmap, vmin=-b, vmax=b,
#                aspect='auto', interpolation='nearest')
#     ax2 = ax1.twinx()
#     ax2.yaxis.tick_left()
#     ax2.plot(A, c='green')
#
#     plt.show()

# def plot_all_layer_heatmap(A,R):
#     for l,a in enumerate(A):
#         if a.ndim == 3: #卷积层
#             a = a[0].detach().numpy().sum(axis=0) #第0维是batch
#             r = np.array(R[l][0]).sum(axis=0)
#             heat_map(a,r)
#         elif a.ndim == 2: #全连接层
#             a = a[0].detach().numpy()
#             r = np.array(R[l][0])
#             heat_map(a, r)

# def plot_batch_heatmap(batch_X,batch_R):
#     '''
#     :param batch_X: batch_size,channel,length
#     :param batch_R: batch_size,channel,length
#     :return:
#     '''
#     if batch_X.ndim == 2:
#         batch_X = batch_X.view(1,batch_X.shape[0],batch_X.shape[1])
#     batch_size = batch_X.shape[0]
#     data = np.array(batch_X.detach().numpy())
#     relevance_score = np.array(batch_R)
#     for i in range(batch_size): # for batch_size
#         for j in range(3): # for channel
#             heat_map(data[i,j,:],relevance_score[i,j,:])


def get_LRP(model,X):
    if X.ndim == 2:
        X = X.view(1,X.shape[0],X.shape[1])
    batch_size = X.shape[0]
    layers1 = list(model._modules['features']._modules['layers'])
    layers2 = list(model._modules['predictor']._modules['layers'])
    L1 = len(layers1)
    L2 = len(layers2)


    # forward and save feature map in A
    A1 = [X] + [None] * L1
    for l in range(L1):
        A1[l + 1] = layers1[l].forward(A1[l])

    feature = A1[-1].view(A1[-1].size(0), -1)
    A2 = [feature] + [None] * L2
    for l in range(L2):
        A2[l + 1] = layers2[l].forward(A2[l])



    # backward and save relevance in R
    T = torch.FloatTensor(np.ones(batch_size)).view(batch_size, 1).to(X.device)
    R2 = [None] * L2 + [(A2[-1] * T).data]

    for l in range(0, L2)[::-1]:  # reverse
        A2[l] = (A2[l].data).requires_grad_(True)

        if isinstance(layers2[l], torch.nn.Linear):
            if l+L1 <= 8:
                rho = lambda p: p + 0.25 * p.clamp(min=0)
                incr = lambda z: z + 1e-9
            if 9 <= l+L1 <= 17:
                rho = lambda p: p
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l+L1 >= 18:
                rho = lambda p: p
                incr = lambda z: z + 1e-9

            z = incr(util.newlayer(layers2[l], rho).forward(A2[l]))  # step 1
            s = (R2[l + 1] / z).data  # step 2
            (z * s).sum().backward()
            c = A2[l].grad  # step 3
            R2[l] = (A2[l] * c).data  # step 4
        else:
            R2[l] = R2[l + 1]

    middle_relevance = R2[0].reshape(A1[-1].shape)
    R1 = [None] * L1 + [middle_relevance]
    for l in range(0, L1)[::-1]:
        A1[l] = (A1[l].data).requires_grad_(True)

        if isinstance(layers1[l], torch.nn.MaxPool1d):  # replace maxpool with avgpool
            layers1[l] = torch.nn.AvgPool1d(2)

        if isinstance(layers1[l], torch.nn.Conv1d) or isinstance(layers1[l], torch.nn.AvgPool1d):
            if l <= 8:
                rho = lambda p: p + 0.25 * p.clamp(min=0)
                incr = lambda z: z + 1e-9
            if 9 <= l <= 17:
                rho = lambda p: p
                incr = lambda z: z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
            if l >= 18:
                rho = lambda p: p
                incr = lambda z: z + 1e-9

            z = incr(util.newlayer(layers1[l], rho).forward(A1[l]))  # step 1
            s = (R1[l + 1] / z).data  # step 2
            (z * s).sum().backward()
            c = A1[l].grad  # step 3
            R1[l] = (A1[l] * c).data  # step 4
        else:
            R1[l] = R1[l + 1]

    return (R1,R2),(A1,A2)  #return all feature maps and relevance scores

def normolize_relevance_score(R,type='minmax'):
    '''
    normalize R to the range of [-1,1] or [0,1]
    :param R: [batch,length]
    :return:
    '''
    assert R.ndim == 2
    re_R = R
    max_r = torch.max(re_R,dim=1)[0].view(R.shape[0],-1)
    min_r = torch.min(re_R,dim=1)[0].view(R.shape[0],-1)

    if type == 'uniform':
        norm_R = (re_R - min_r) / (max_r - min_r)
    elif type == 'minmax':
        norm_R = 2 * ((re_R - min_r) / (max_r - min_r)) - 1
    else:
        raise ValueError('Error!')

    return_R = norm_R.view(R.shape)
    return return_R

def get_LRP_weigth(R,type='minmax',scaler=0.5):
    init_weight = torch.ones(R.shape).to(R.device)
    LRP_weight = scaler * normolize_relevance_score(R,type) + init_weight
    return LRP_weight


if __name__ == '__main__':
    pass


