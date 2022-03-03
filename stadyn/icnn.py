import math
import torch
import torch.nn as nn


class ICNN(nn.Module):
    """Input-convex neural network.
    """
    def __init__(self, dim_in:int, dims_hidden:list,
                 activation:str='elu', batchnorm:bool=False):
        super(ICNN, self).__init__()

        dims_all = [dim_in,] + dims_hidden + [1,]

        nonneg_weights = []
        for i in range(len(dims_all)-1): # k-1 layers
            nonneg_weights.append(nn.Parameter(torch.zeros(dims_all[i], dims_all[i+1])))
        self.nonneg_weights = nn.ParameterList(nonneg_weights)
        for w in self.nonneg_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        linear_modules = []
        for i in range(len(dims_all)): # k layers
            linear_modules.append(nn.Linear(dims_all[0], dims_all[i]))
        self.linear_modules = nn.ModuleList(linear_modules)

        bn_modules = []
        for i in range(len(dims_all)-1):
            if batchnorm:
                bn_modules.append(nn.BatchNorm1d(dims_all[i]))
            else:
                bn_modules.append(nn.Sequential())
        self.bn_modules = nn.ModuleList(bn_modules)

        act_modules = []
        for i in range(len(dims_all)-1):
            if activation == 'softplus':
                act_modules.append(nn.Softplus())
            elif activation == 'relu':
                act_modules.append(nn.ReLU())
            elif activation == 'leakyrelu':
                act_modules.append(nn.LeakyReLU())
            elif activation == 'prelu':
                act_modules.append(nn.PReLU())
            elif activation == 'elu':
                act_modules.append(nn.ELU())
            else:
                raise ValueError('unknown activation function specified')
        self.act_modules = nn.ModuleList(act_modules)

        self.dim_in = dim_in

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, self.dim_in)
        out = self.linear_modules[0](z_)
        out = self.bn_modules[0](out)
        out = self.act_modules[0](out)
        for i, nonneg_weight in enumerate(self.nonneg_weights):
            U = torch.nn.functional.softplus(nonneg_weight)
            out = out@U/U.shape[0] + self.linear_modules[i+1](z_)
            if i < len(self.nonneg_weights)-1:
                out = self.bn_modules[i+1](out)
                out = self.act_modules[i+1](out)
        return out
