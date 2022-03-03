import torch
import torch.nn as nn

from stadyn.anode import ODEBlock
from stadyn.icnn import ICNN
from stadyn.mlp import MLP


class SReLU1(nn.Module):
    """Smoothed ReLU function in C^1.
    """
    def __init__(self, threshold:float=1.0):
        super(SReLU1, self).__init__()
        assert threshold > 0.0
        self.threshold = threshold

    def forward(self, x):
        out = torch.where(x<=0.0, torch.zeros_like(x, device=x.device), x)
        out = torch.where(out<self.threshold, out*out/2.0/self.threshold, out-self.threshold/2.0)
        return out


class CandFun(nn.Module):
    """Lyapunov candidate function.
    """
    def __init__(self, invset:nn.Module, invset_mode:str, dim_z:int, dim_invset:int,
                 dims_hidden_warpfun:list, dims_hidden_convexfun:list,
                 dim_aug:int=0, activation:str='elu', batchnorm:bool=False,
                 eps:float=0.1, warping:bool=False):
        super(CandFun, self).__init__()
        assert invset_mode=='vol' or invset_mode=='surf'
        assert dim_invset <= dim_z
        self.invset = invset
        self.invset_mode = invset_mode
        self.dim_z = dim_z
        self.dim_invset = dim_invset
        self.eps = eps
        self.warping = warping

        self.warpfun = ODEBlock(dim_z, dim_aug, dims_hidden_warpfun)
        self.convexfun = ICNN(dim_z, dims_hidden_convexfun, activation=activation, batchnorm=batchnorm)
        self.lastact = SReLU1()

    def forward(self, z):
        z_ = z.view(-1, self.dim_z)
        batchsize = z_.shape[0]

        z_proj = self.invset.proj(z_[:,:self.dim_invset], self.invset_mode)
        if self.dim_invset < self.dim_z:
            z_proj = torch.cat([z_proj, torch.zeros((batchsize, self.dim_z-self.dim_invset), device=z_.device)], dim=1)
        if self.warping:
            z_warped = self.warpfun(z_)[:,:self.dim_z]
            z_proj_warped = self.warpfun(z_proj)[:,:self.dim_z]
        else:
            z_warped = z_
            z_proj_warped = z_proj
        value = self.convexfun(z_warped) - self.convexfun(z_proj_warped)
        value = self.lastact(value) + self.eps*torch.sum(torch.pow(z_-z_proj, 2), dim=1).view(-1,1)

        return value.view(-1)
