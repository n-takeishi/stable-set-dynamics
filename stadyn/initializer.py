import torch
import torch.nn as nn

from stadyn.mlp import MLP

def _sliding_window(x:torch.Tensor, window_size:int, stride:int=1):
    return x.unfold(0, window_size, stride).permute(0,2,1)

class Initializer(nn.Module):
    """Initializer.
    """
    def __init__(self, dim_x:int, lag:int, dims_hidden:list=None,
                 activation:str='elu', dropout:float=-1.0,
                 batchnorm:bool=False, binary:bool=False):
        super(Initializer, self).__init__()
        self.dim_x = dim_x
        self.lag = lag

        if dims_hidden is None:
            dims_hidden = [int((dim_x*lag+dim_x)/2),]

        if lag>0:
            self.net = MLP([dim_x*lag,]+dims_hidden+[dim_x,], activation=activation,
                           dropout=dropout, batchnorm=batchnorm, binary=binary)
        else:
            self.net = nn.Sequential()

    def forward(self, x:torch.Tensor):
        # x_{t-k+1}, ..., x_t --> init_x_t
        x_ = x.view(-1, self.dim_x)
        if self.lag>0:
            dim_in = self.dim_x * self.lag
            init_x = self.net(_sliding_window(x_, self.lag).view(-1, dim_in))
            return init_x # shape=(batchsize-lag+1, dim_x)
        else:
            return x_
