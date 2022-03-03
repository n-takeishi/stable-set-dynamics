"""
Implementation from https://github.com/EmilienDupont/augmented-neural-odes

MIT License

Copyright (c) 2019 Emilien Dupont

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from stadyn.mlp import MLP

METHOD = 'rk4'
ACTIVATION = 'softplus'
MAX_NUM_STEPS = 1000
RTOL = 1e-3
ATOL = 1e-3
ADJOINT = False

if METHOD=='dopri5':
    OPTIONS = {'max_num_steps':MAX_NUM_STEPS}
else:
    OPTIONS = {}


class _ODEFunc(nn.Module):
    """MLP modeling the derivative of ODE system.
    """
    def __init__(self, dim_x:int, dim_aug:int, dims_hidden:list):
        super(_ODEFunc, self).__init__()

        dims_all = [dim_x+dim_aug,] + dims_hidden + [dim_x+dim_aug,]
        self.mlp = MLP(dims_all, activation=ACTIVATION)
        self.nfe = 0
        self.reverse = False

    def forward(self, t:torch.Tensor, x:torch.Tensor):
        self.nfe += 1
        out = self.mlp(x)
        if self.reverse:
            return -out
        else:
            return out


class ODEBlock(nn.Module):
    """Solves ODE.
    """
    def __init__(self, dim_x:int, dim_aug:int, dims_hidden:list):
        super(ODEBlock, self).__init__()
        self.dim_x = dim_x
        self.dim_aug = dim_aug

        self.odefunc = _ODEFunc(dim_x, dim_aug, dims_hidden)

    def forward(self, x:torch.Tensor):
        self.odefunc.nfe = 0
        self.odefunc.reverse = False

        batchsize = int(x.numel()/self.dim_x)
        x_aug = x.view(-1, self.dim_x).clone()
        if self.dim_aug > 0:
            aug = torch.zeros((batchsize, self.dim_aug), device=x.device)
            x_aug = torch.cat([x_aug, aug], dim=1)

        integration_time = torch.tensor([0, 1]).float().type_as(x)
        if ADJOINT:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=RTOL, atol=ATOL, method=METHOD, options=OPTIONS)
        else:
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=RTOL, atol=ATOL, method=METHOD, options=OPTIONS)

        # return augmented result
        return out[1]

    def reverse(self, x_aug:torch.Tensor):
        self.odefunc.nfe = 0
        self.odefunc.reverse = True

        integration_time = torch.tensor([0, 1]).float().type_as(x_aug)
        x_aug_ = x_aug.view(-1, self.dim_x+self.dim_aug).clone()
        if ADJOINT:
            out = odeint_adjoint(self.odefunc, x_aug_, integration_time,
                                 rtol=RTOL, atol=ATOL, method=METHOD, options=OPTIONS)
        else:
            out = odeint(self.odefunc, x_aug_, integration_time,
                         rtol=RTOL, atol=ATOL, method=METHOD, options=OPTIONS)

        self.odefunc.reverse = False

        # return augmented result
        return out[1]
