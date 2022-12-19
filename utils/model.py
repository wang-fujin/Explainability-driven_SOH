import torch
import torch.nn as nn

class backbone(nn.Module):
    def __init__(self):
        super(backbone,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),  # b,3,128 -> b,64,128
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),  # b,64,128
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # b,64,64

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # b,64,64 -> b,128,64
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # b,128,32

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # b,256,16

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # b,512,8

            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # b,256,4
        )

    def forward(self,x):
        out = self.layers(x)
        out = torch.flatten(out, 1)

        return out


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
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



if __name__ == '__main__':

    m2 = model()
    print(m2)
    x = torch.rand(5,3,128)
    y1 = m2(x)
    y2 = m2.forward(x)

    print(y1)
    print(y2)
    # print(m1._modules['features'])
    # print(m2._modules['features']._modules['layers'])
    # print()
    # print(m1._modules['predictor'])
    # print(m2._modules['predictor']._modules['layers'])