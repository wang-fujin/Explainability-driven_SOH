import torch
import torch.nn as nn
import numpy as np
from utils import util

class backbone(nn.Module):
    def __init__(self,input_size=3):
        super(backbone,self).__init__()

        self.layers = nn.LSTM(input_size=input_size,hidden_size=128,num_layers=2,batch_first=True)


    def forward(self,x):
        out,(h,c) = self.layers(x)
        return out[:,-1,:]


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(64, 1),
            # nn.Sigmoid()，
        )
    def forward(self,x,weight=None):
        if weight is not None:
            x = torch.mul(x, weight.detach())
        out = self.layers(x)
        return out

class model(nn.Module):
    def __init__(self,init_weights=True):
        super(model,self).__init__()
        self.features = backbone()
        self.predictor = Predictor()
        if init_weights:
            self._initialize_weights()
    def forward(self,x,weight=None):
        x = x.transpose(1, 2)
        embed = self.features(x)
        pred = self.predictor(embed,weight)
        return pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def get_LRP(model,X):
    if X.ndim == 2:
        X = X.view(1,X.shape[0],X.shape[1])
    batch_size = X.shape[0]
    lstm = model._modules['features']._modules['layers']
    layers2 = list(model._modules['predictor']._modules['layers'])

    L2 = len(layers2)


    # forward and save feature map in A
    X = X.transpose(1, 2)
    out,_ = lstm.forward(X)
    feature = out[:,-1,:]


    A2 = [feature] + [None] * L2
    for l in range(L2):
        A2[l + 1] = layers2[l].forward(A2[l])

    # backward and save relevance in R
    T = torch.FloatTensor(np.ones(batch_size)).view(batch_size, 1).to(X.device)
    R2 = [None] * L2 + [(A2[-1] * T).data]

    for l in range(0, L2)[::-1]:  # reverse
        A2[l] = (A2[l].data).requires_grad_(True)

        if isinstance(layers2[l], torch.nn.Linear):

            rho = lambda p: p
            incr = lambda z: z + 1e-9

            z = incr(util.newlayer(layers2[l], rho).forward(A2[l]))  # step 1
            s = (R2[l + 1] / z).data  # step 2
            (z * s).sum().backward()
            c = A2[l].grad  # step 3
            R2[l] = (A2[l] * c).data  # step 4
        else:
            R2[l] = R2[l + 1]

    return R2


if __name__ == '__main__':
    batch_size = 32
    seq_len = 128
    input_size = 3
    x = torch.randn(batch_size,seq_len,input_size)
    net = model()
    y = net(x)
    print(f'input:{x.shape}, output:{y.shape}')

    relevance = get_LRP(net,x)
    print('relevance shape:',relevance[0].shape)
