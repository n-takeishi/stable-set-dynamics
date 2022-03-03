import torch
import torch.nn as nn

class MLP(nn.Module):
    """Multi-layer perceptron.
    """
    def __init__(self, dims_all:list, activation:str='elu',
                 dropout:float=-1.0, batchnorm:bool=False, binary:bool=False):
        super(MLP, self).__init__()

        modules = []

        # from first to second-last layer
        for i in range(len(dims_all)-2):
            # fully-connected
            modules.append(nn.Linear(dims_all[i], dims_all[i+1]))
            # batch normalization if any
            if batchnorm:
                modules.append(nn.BatchNorm1d(dims_all[i+1]))
            # nonlinear activation
            if activation == 'softplus':
                modules.append(nn.Softplus())
            elif activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'leakyrelu':
                modules.append(nn.LeakyReLU())
            elif activation == 'prelu':
                modules.append(nn.PReLU())
            elif activation == 'elu':
                modules.append(nn.ELU())
            else:
                raise ValueError('unknown activation function specified')
            # dropout if any
            if dropout>0.0:
                modules.append(nn.Dropout(p=dropout))

        # last layer
        modules.append(nn.Linear(dims_all[-2], dims_all[-1]))
        if binary:
            modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*modules)
        self.dim_in = dims_all[0]

    def forward(self, x:torch.Tensor):
        out = self.net(x.view(-1, self.dim_in))
        return out
