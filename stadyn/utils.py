import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def loaddata(datadir:str, datafile:str,
             device:torch.device='cpu', dotY:bool=True):
    if datafile.endswith('.mat'):
        mat = loadmat(datadir+'/'+datafile)
        # (num_episodes, len_episode)
        data_T = torch.tensor(mat['T'], dtype=torch.float).to(device)
        # (num_episodes, len_episode, dim_x)
        data_Y = torch.tensor(mat['Y'], dtype=torch.float).to(device)
        if dotY:
            # (num_episodes, len_episode, dim_x)
            data_dotY = torch.tensor(mat['dotY'], dtype=torch.float).to(device)
    else:
        raise ValueError('unknown data form')

    if dotY:
        return data_T, data_Y, data_dotY
    else:
        return data_T, data_Y


def set_bn_eval(m:nn.Module):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m:nn.Module):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()

def plot_flow_field(ax, f, u_range, v_range, n_grid=None, quiver=False):
    # code from http://be150.caltech.edu/2017/handouts/dynamical_systems_approaches.html

    if n_grid is None:
        if quiver:
            n_grid = 10
        else:
            n_grid = 100

    # Set up u,v space
    u = np.linspace(u_range[0], u_range[1], n_grid)
    v = np.linspace(v_range[0], v_range[1], n_grid)
    uu, vv = np.meshgrid(u, v)

    # Compute derivatives
    u_vel = np.empty_like(uu)
    v_vel = np.empty_like(vv)
    for i in range(uu.shape[0]):
        for j in range(uu.shape[1]):
            [[u_vel[i,j]], [v_vel[i,j]]] = f(np.array([[uu[i,j]], [vv[i,j]]]))

    # Compute speed
    speed = np.sqrt(u_vel**2 + v_vel**2)

    # Make linewidths proportional to speed,
    # with minimal line width of 0.5 and max of 3
    lw = 0.5 + 2.5 * speed / speed.max()

    # Make stream plot
    if quiver:
        ax.quiver(uu, vv, u_vel, v_vel)
    else:
        ax.streamplot(uu, vv, u_vel, v_vel, linewidth=lw, arrowsize=1.2, density=1, color='thistle')

    return uu, vv, u_vel, v_vel
