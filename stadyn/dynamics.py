import math
import torch
import torch.nn as nn
from torch.autograd import grad

from stadyn.anode import ODEBlock
from stadyn.mlp import MLP
from stadyn.geometry import *
from stadyn.lyapunov import CandFun
from stadyn.utils import *


def _is_almost_zero(x, eps:float):
    if type(x) is torch.Tensor:
        return torch.abs(x)<eps
    else:
        raise NotImplementedError()


def _smooth_step(x, b):
    assert b>=0
    device = x.device
    if type(b) is torch.Tensor:
        tmp = 2.0 / torch.pow(b,2)
    elif type(b) is float:
        tmp = 2.0 / b**2
    else:
        raise ValueError()

    out1 = torch.where( 0.0<=x           , torch.ones_like(x,device=device), x )
    out2 = torch.where( (-b/2.0<=x) & (x<0.0), -tmp*torch.pow(out1,2)+1.0, out1 )
    out3 = torch.where( (-b<=x) & (x<-b/2.0) , tmp*torch.pow(out2+b,2), out2 )
    out4 = torch.where( x<-b             , torch.zeros_like(out3,device=device), out3 )
    return out4


class _Slack(nn.Module):
    def __init__(self, dim_z:int, dims_hidden:list,
        activation:str='elu', batchnorm:bool=False):
        super(_Slack, self).__init__()
        self.dim_z = dim_z
        self.fun = MLP([dim_z,]+dims_hidden+[1,],
            activation=activation, dropout=-1.0, batchnorm=batchnorm, binary=False)

    def forward(self, z:torch.Tensor):
        return torch.clamp(self.fun(z.view(-1, self.dim_z)), min=0.0).view(-1)


class _Zero(nn.Module):
    def __init__(self, dim_z:int):
        super(_Zero, self).__init__()
        self.dim_z = dim_z

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, self.dim_z)
        batchsize = z_.shape[0]
        return torch.zeros((batchsize,), device=z_.device)


class Dynamics(nn.Module):
    def __init__(self, dim_x:int, dim_invset:int=None,
                 invset_type:str='sphere', invset_mode:str='vol', invset_learnable:bool=False,
                 invset_sphere_init:float=1.0,
                 transform:bool=True, invariance:bool=True, stability:bool=True, alpha:float=0.05,
                 dims_hidden_tran:list=None, dims_hidden_base:list=None,
                 dims_hidden_lyap_warping:list=None, dims_hidden_lyap_convexfun:list=None,
                 dim_aug_tran:int=0, dim_aug_lyap_warping:int=0,
                 eps:float=0.1, warping:bool=False,
                 activation:str='elu', dropout:float=-1.0, batchnorm:bool=False,
                 zero_slack:bool=False, smooth_step:float=-1.0):
        super(Dynamics, self).__init__()
        assert dim_invset <= dim_x
        if dim_invset is None:
            dim_invset = dim_x

        self.dim_x = dim_x
        self.dim_z = dim_x; dim_z = dim_x
        self.dim_invset = dim_invset
        self.transform = transform
        self.invariance = invariance
        self.stability = stability

        # invariant set
        if invset_type == 'sphere':
            self.invset = Sphere(dim_invset, invset_sphere_init, learnable=invset_learnable)
        elif invset_type == 'quadric':
            self.invset = Quadric(dim_invset, init_sqrtA=None, init_b=None, init_c=None)
        elif invset_type == 'linear':
            self.invset = SimpleLinear(dim_invset, init_u=None, learnable=invset_learnable)
        elif invset_type == 'optlinear':
            self.invset = Linear(dim_invset, init_a=None, init_b=None, bias=False)

        # invertible feature transform x --> z
        if dims_hidden_tran is None:
            dims_hidden_tran = [dim_x,]
        self.phi = ODEBlock(dim_x, dim_aug_tran, dims_hidden_tran)

        # base dynamics z --> dz/dt
        if dims_hidden_base is None:
            dims_hidden_base = [dim_z,]
        self.h = MLP([dim_z,]+dims_hidden_base+[dim_z,],
            activation=activation, dropout=dropout, batchnorm=batchnorm, binary=False)

        # Lyapunov function z --> R
        if dims_hidden_lyap_warping is None:
            dims_hidden_lyap_warping = [dim_z,]
        if dims_hidden_lyap_convexfun is None:
            dims_hidden_lyap_convexfun = [dim_z,]
        self.V = CandFun(self.invset, invset_mode, dim_z, dim_invset,
            dims_hidden_lyap_warping, dims_hidden_lyap_convexfun, dim_aug_lyap_warping,
            activation=activation, batchnorm=False, eps=eps, warping=warping)

        # unit step function
        if smooth_step>0.0:
            self.gamma = lambda y: _smooth_step(y, smooth_step)
        else:
            self.gamma = lambda y: torch.where(y>0.0,
                torch.ones_like(y, device=y.device), torch.zeros_like(y, device=y.device))

        # impulse-like indicator function (with tiny tolerance)
        self.delta = lambda y: torch.where(_is_almost_zero(y, 1e-3),
            torch.ones_like(y, device=y.device), torch.zeros_like(y, device=y.device))

        # the coefficient in the Lyapunov condition
        self.alpha = alpha

        # slack function for asymptotic stability
        if zero_slack:
            self.eta = _Zero(dim_z)
        else:
            self.eta = _Slack(dim_z, [dim_z*2,], activation=activation, batchnorm=False)

        # slack function for invariance
        if invset_mode == 'vol':
            self.xi = _Slack(dim_z, [dim_z*2,], activation=activation, batchnorm=False)
        elif invset_mode == 'surf':
            self.xi = _Zero(dim_z)

        self.detach = False

    def forward(self, t:torch.Tensor, x:torch.Tensor, return_V:str=None):
        x_ = x.view(-1, self.dim_x); batchsize = x_.shape[0]

        if self.transform:
            z_aug = self.phi(x_)
            z = z_aug[:, :self.dim_z]
            aug = z_aug[:, self.dim_z:]
        else:
            z = x_.clone().requires_grad_()

        h = self.h(z)

        g = h
        if self.stability:
            nz1 = self.V(z).detach().nonzero().view(-1)
            eta = self.eta(z)
            if nz1.shape[0] > 0:
                if self.training:
                    self.V.apply(set_bn_eval)

                z_nz1 = z[nz1]
                V_nz1 = self.V(z_nz1)
                grad_V_nz1 = grad([v for v in V_nz1], [z_nz1], create_graph=self.training, only_inputs=True)[0]
                norm_grad_V_nz1 = torch.sum(torch.pow(grad_V_nz1,2), dim=1).clamp(min=1e-9)
                cond_nz1 = torch.sum(grad_V_nz1*h[nz1], 1) + self.alpha*V_nz1

                gamma_cond_nz1 = self.gamma(cond_nz1)

                nz2 = gamma_cond_nz1.detach().nonzero().view(-1)
                if nz2.shape[0] > 0:
                    adjuster1_nz2 = gamma_cond_nz1[nz2] / norm_grad_V_nz1[nz2] * (cond_nz1[nz2] + eta[nz1[nz2]])
                    adjuster1_nz2 = adjuster1_nz2.view(-1,1) * grad_V_nz1[nz2]
                    g[nz1[nz2]] = g[nz1[nz2]] - adjuster1_nz2

                if self.training:
                    self.V.apply(set_bn_train)

        f = g
        if self.invariance:
            remnorm = torch.sum(torch.pow(z[:,self.dim_invset:], 2), dim=1)
            nz = self.delta(self.invset(z[:,:self.dim_invset]) + remnorm).detach().nonzero().view(-1)
            xi = self.xi(z)
            if nz.shape[0] > 0:
                z_nz = z[nz,:self.dim_invset]
                C_nz = self.invset(z_nz)
                grad_C_nz = grad([c for c in C_nz], [z_nz], create_graph=self.training, only_inputs=True)[0]
                norm_grad_C_nz = torch.sum(torch.pow(grad_C_nz,2), dim=1).clamp(min=1e-9)

                remnorm = torch.sum(torch.pow(z[nz,self.dim_invset:], 2), dim=1)
                delta_C_nz = self.delta(C_nz + remnorm)

                adjuster2_nz = delta_C_nz / norm_grad_C_nz * (torch.sum(grad_C_nz*g[nz,:self.dim_invset], 1) - xi[nz])
                adjuster2_nz = adjuster2_nz.view(-1,1) * grad_C_nz
                f[nz,:self.dim_invset] = f[nz,:self.dim_invset] - adjuster2_nz

        if self.transform:
            f_aug = torch.cat([f, aug], dim=1)
            out_aug = self.phi.reverse(f_aug)
            out = out_aug[:, :self.dim_z]
        else:
            out = f

        if self.detach:
            out = out.detach()

        if return_V is None:
            return out
        elif self.stability and nz1.shape[0]>0:
            if return_V == 'mean':
                return out, torch.mean(V_nz1)
            elif return_V == 'max':
                return out, torch.max(V_nz1)

        return out, torch.zeros(1,device=x_.device)
