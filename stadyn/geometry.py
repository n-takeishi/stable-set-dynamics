import math
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class Sphere(nn.Module):
    """ I = sphere centered at z=0
    """
    def __init__(self, dim_invset:int, radius:float,
                 learnable:bool=False):
        super(Sphere, self).__init__()
        assert radius >= 0.0
        self.dim_invset = dim_invset
        self.learnable = learnable
        if learnable:
            self.radius = nn.Parameter(torch.ones(1, requires_grad=True)*radius)
        else:
            self.register_buffer('radius', torch.ones(1)*radius)

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, self.dim_invset)
        z_norm_sq = torch.sum(torch.pow(z_, 2), dim=1)
        return self.radius*self.radius - z_norm_sq

    def proj(self, z:torch.Tensor, mode:str):
        z_ = z.view(-1, self.dim_invset); batchsize = z_.shape[0]
        z_norm_sq = torch.sum(torch.pow(z_, 2), dim=1).view(-1,1)
        z_norm = torch.sqrt(z_norm_sq)

        C = self.radius*self.radius - z_norm_sq
        C = C.detach()

        if mode=='surf' or self.radius==0.0:
            z_norm_ = torch.where(z_norm==0.0, torch.ones_like(z_norm,device=z_.device), z_norm)
            z_proj = z_ / z_norm_ * self.radius
        elif mode=='vol' and self.radius!=0.0:
            out = (C<0.0).nonzero()
            z_proj_out = z_[out] / z_norm[out].clamp(min=1e-9) * self.radius
            z_proj = z_.clone()
            z_proj[out] = z_proj_out
        else:
            raise RuntimeError('projection error')

        return z_proj


class TwoTorus(nn.Module):
    """ I = 2-torus centered at z=0 (so dim_invset=3 only)
    """
    def __init__(self,  radius_main:float, radius_tube:float,
                 learnable:bool=False):
        super(TwoTorus, self).__init__()
        assert radius_tube > 0.0
        assert radius_main > radius_tube
        self.learnable = learnable
        if learnable:
            self.radius_main = nn.Parameter(torch.ones(1, requires_grad=True)*radius_main)
            self.radius_tube = nn.Parameter(torch.ones(1, requires_grad=True)*radius_tube)
        else:
            self.register_buffer('radius_main', torch.ones(1)*radius_main)
            self.register_buffer('radius_tube', torch.ones(1)*radius_tube)

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, 3)
        norm_01 = torch.sqrt(torch.pow(z_[:,0],2) + torch.pow(z_[:,1],2))
        norm_2_sq = torch.pow(z_[:,2],2)
        return self.radius_tube*self.radius_tube - (torch.pow(self.radius_main-norm_01, 2) + norm_2_sq)

    def proj(self, z:torch.Tensor, mode:str):
        z_ = z.view(-1, 3); batchsize = z_.shape[0]
        norm_01 = torch.sqrt(torch.pow(z_[:,0],2) + torch.pow(z_[:,1],2))
        norm_2_sq = torch.pow(z_[:,2],2)

        coeff_2 = self.radius_tube / torch.sqrt(torch.pow(self.radius_main-norm_01, 2) + norm_2_sq)
        coeff_01 = self.radius_main / norm_01 + (1.0 - self.radius_main / norm_01)*coeff_2

        C = self.radius_tube*self.radius_tube - (torch.pow(self.radius_main-norm_01, 2) + norm_2_sq)
        C = C.detach()

        z_proj_ = z_ * torch.cat([coeff_01, coeff_01, coeff_2], dim=1)
        z_proj = z_.clone()
        for i in range(batchsize):
            if (mode=='vol' and C[i]<0.0) or (mode=='surf' and C[i]!=0.0):
                # if not in I, and
                if norm_01[i] == 0.0:
                    # if on the center axis, then choose an arbitrary point from the axis-oriented circle
                    z_proj[i][0] = self.radius_main * (1.0 - coeff_2[i])
                    z_proj[i][1] = 0.0
                    z_proj[i][2] = z_[i][2] * coeff_2[i]
                elif norm_01[i] == self.radius_main and norm_2_sq[i] == 0.0:
                    # if on the major circle, then choose an arbitrary point from the tube-oriented circle
                    z_proj[i][0] = z_[i][0] * (self.radius_tube/self.radius_main + 1.0)
                    z_proj[i][1] = z_[i][1] * (self.radius_tube/self.radius_main + 1.0)
                    z_proj[i][2] = 0.0
                else:
                    # otherwise, then project
                    z_proj[i] = z_proj_[i]

        return z_proj


class SimpleLinear(nn.Module):
    def __init__(self,  dim_invset:int, init_u:torch.Tensor=None, learnable:bool=True):
        super(SimpleLinear, self).__init__()
        self.dim_invset = dim_invset

        if init_u is None:
            init_u = torch.rand(dim_invset, requires_grad=True)*2.0-1.0

        if learnable:
            self.u = nn.Parameter(init_u)
        else:
            self.register_buffer('u', init_u)

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, self.dim_invset)
        out = torch.sum(z_*self.u, dim=1)
        out = -out
        return out

    def proj(self, z:torch.Tensor, mode:str):
        z_ = z.view(-1, self.dim_invset); batchsize = z_.shape[0]

        C = self.forward(z_).detach()

        u_norm_sq = torch.sum(torch.pow(self.u,2))

        if mode=='vol':
            nz = (C<0.0).nonzero().view(-1)
            z_proj = z_.clone()
            if nz.shape[0]>0:
                z_proj_nz = torch.ones((nz.shape[0], self.dim_invset), device=z_.device)
                z_proj_nz = z_proj_nz * (torch.sum(z_[nz]*self.u, dim=1)/u_norm_sq).view(nz.shape[0], 1)
                z_proj_nz = z_proj_nz * self.u
                z_proj[nz] = z_proj_nz
        elif mode=='surf':
            z_proj = torch.ones((batchsize, self.dim_invset), device=z_.device)
            z_proj = z_proj * (torch.sum(z_*self.u, dim=1)/u_norm_sq).view(batchsize, 1)
            z_proj = z_proj * self.u

        return z_proj


class Linear(nn.Module):
    """ I = {z | C(z) >= 0}
    -C(z) = a^T z + b
    """
    def __init__(self,  dim_invset:int,
                 init_a:torch.Tensor=None, init_b:torch.Tensor=None, bias:bool=False):
        super(Linear, self).__init__()
        self.dim_invset = dim_invset

        if init_a is None:
            init_a = torch.rand(dim_invset, requires_grad=True)*2.0-1.0
        if init_b is None:
            init_b = -1.0 * torch.rand(1, requires_grad=True)*2.0-1.0

        self.a = nn.Parameter(init_a)
        if bias:
            self.b = nn.Parameter(init_b)
        else:
            self.register_buffer('b', torch.zeros(1))

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, self.dim_invset)
        out = torch.sum(z_*self.a, dim=1) + self.b
        out = -out
        return out

    def proj(self, z:torch.Tensor, mode:str):
        z_ = z.view(-1, self.dim_invset); batchsize = z_.shape[0]

        C = self.forward(z_).detach()

        _a = cp.Parameter(self.dim_invset)
        _b = cp.Parameter(1)
        _z = cp.Parameter(self.dim_invset)
        _z_proj = cp.Variable(self.dim_invset)

        obj = cp.Minimize(0.5*cp.sum_squares(_z_proj-_z))
        if mode=='vol':
            cons = [-cp.sum(cp.multiply(_a, _z_proj)) - _b >= 0.0]
        elif mode=='surf':
            cons = [-cp.sum(cp.multiply(_a, _z_proj)) - _b == 0.0]

        prob = cp.Problem(obj, cons)
        layer = CvxpyLayer(prob, parameters=[_a, _b, _z], variables=[_z_proj])

        if mode=='vol':
            idx = (C<0.0).nonzero()
        elif mode=='surf':
            idx = (C!=0.0).nonzero()

        z_proj = z_.clone()
        for i in idx:
            z_proj_i = layer(self.a, self.b, z_[i].view(self.dim_invset))
            z_proj[i] = z_proj_i[0].clone()

        return z_proj


class Quadric(nn.Module):
    """ I = {z | C(z) >= 0}
    -C(z) = 0.5*z^T A z + b^T z + c
    """
    def __init__(self, dim_invset:int, init_sqrtA:torch.Tensor=None,
                 init_b:torch.Tensor=None, init_c:torch.Tensor=None):
        super(Quadric, self).__init__()
        self.dim_invset = dim_invset

        if init_sqrtA is None:
            init_sqrtA = torch.rand([dim_invset,dim_invset], requires_grad=True)
        if init_b is None:
            init_b = torch.rand(dim_invset, requires_grad=True)*2.0-1.0
        if init_c is None:
            init_c = torch.rand(1, requires_grad=True)*2.0-1.0

        self.sqrtA = nn.Parameter(init_sqrtA)
        self.b = nn.Parameter(init_b)
        self.c = nn.Parameter(init_c)

    def forward(self, z:torch.Tensor):
        z_ = z.view(-1, self.dim_invset)
        out = 0.5*torch.sum(torch.pow(z_@self.sqrtA.T, 2), dim=1) + torch.sum(z_*self.b, dim=1) + self.c
        out = -out
        return out

    def proj(self, z:torch.Tensor, mode:str):
        z_ = z.view(-1, self.dim_invset); batchsize = z_.shape[0]

        C = self.forward(z_).detach()

        _sqrtA = cp.Parameter((self.dim_invset, self.dim_invset))
        _b = cp.Parameter(self.dim_invset)
        _c = cp.Parameter(1)
        _z = cp.Parameter(self.dim_invset)
        _z_proj = cp.Variable(self.dim_invset)

        obj = cp.Minimize(0.5*cp.sum_squares(_z_proj-_z))
        if mode=='vol':
            cons = [-0.5*cp.sum_squares(cp.matmul(_sqrtA, _z_proj))
                        - cp.sum(cp.multiply(_b, _z_proj)) - _c >= 0.0]
        elif mode=='surf':
            raise NotImplementedError()
            cons = [-0.5*cp.sum_squares(cp.matmul(_sqrtA, _z_proj))
                        - cp.sum(cp.multiply(_b, _z_proj)) - _c == 0.0]
            # NOTE: This is not DCP, but DCCP, so cvxpylayers cannot handle it :(

        prob = cp.Problem(obj, cons)
        layer = CvxpyLayer(prob, parameters=[_sqrtA, _b, _c, _z], variables=[_z_proj])

        if mode=='vol':
            idx = (C<0.0).nonzero()
        elif mode=='surf':
            idx = (C!=0.0).nonzero()

        z_proj = z_.clone()
        for i in idx:
            z_proj_i = layer(self.sqrtA, self.b, self.c, z_[i].view(self.dim_invset))
            z_proj[i] = z_proj_i[0].clone()

        return z_proj
