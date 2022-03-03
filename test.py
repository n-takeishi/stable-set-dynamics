import argparse, json
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
from torchdiffeq import odeint, odeint_adjoint

from stadyn.utils import *
import models

#--------------------------
# Configuration
#--------------------------

parser = argparse.ArgumentParser(description='')

# input
parser.add_argument('--inprefix', type=str, required=True)
parser.add_argument('--indir', type=str, required=True)
parser.add_argument('--argsfile', type=str, default='!INPREFIX!_args_single.json')
parser.add_argument('--modelfile-system', type=str, default='!INPREFIX!_system_single.pt')
parser.add_argument('--modelfile-initer', type=str, default='')
parser.add_argument('--datadir', type=str, default='./')
parser.add_argument('--datafile-te', type=str, default='data_te.mat')
# output
parser.add_argument('--outprefix', type=str, required=True)
parser.add_argument('--outdir', type=str, required=True)
# mode
parser.add_argument('--plot-vf', action='store_true', default=False)
parser.add_argument('--plot-traj', action='store_true', default=False)
parser.add_argument('--plot-lyap', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
# plot
parser.add_argument('--plot-range-x', type=float, default=2.0)
parser.add_argument('--plot-range-y', type=float, default=2.0)
# test/generate
parser.add_argument('--testepi', type=int, nargs='+', default=[0])
parser.add_argument('--testplotfeat', type=int, default=-1)
parser.add_argument('--testsavepred', action='store_true', default=False)
parser.add_argument('--testsavedif', action='store_true', default=False)
parser.add_argument('--geninit', type=float, nargs='+', default=[0.0, 0.0])
parser.add_argument('--gentend', type=float, default=10.0)
parser.add_argument('--genlen', type=int, default=100)
# others
parser.add_argument('--normdy', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1234567890)
parser.add_argument('--no-cuda', action='store_true', default=False)

args_te = vars(parser.parse_args())

outprefix = args_te['outdir']+'/'+args_te['outprefix']+'_'

# set device
device = torch.device('cuda' if not args_te['no_cuda'] and torch.cuda.is_available() else 'cpu')

# replace !INPREFIX!
for key in args_te:
    if type(args_te[key]) is str:
        args_te[key] = args_te[key].replace('!INPREFIX!', args_te['inprefix'])

# set some other variables
plot_range_x = [-args_te['plot_range_x'], args_te['plot_range_x']]
plot_range_y = [-args_te['plot_range_y'], args_te['plot_range_y']]

#--------------------------
# Load models
#--------------------------

np.random.seed(args_te['seed']); torch.manual_seed(args_te['seed'])


# load original (training-time) args
argspath = args_te['indir']+'/'+args_te['argsfile']
with open(argspath, 'r') as fp:
    args_tr = json.load(fp)

# load models
system, encoder, decoder, initer = models.new(device, args_tr)

models.load(args_te['indir'],
    modelfile_system=args_te['modelfile_system'],
    modelfile_initer=args_te['modelfile_initer'],
    system=system, initer=initer, disp=True)

system.eval(); encoder.eval(); decoder.eval(); initer.eval()
system.detach=True

#--------------------------
# Plot vector field
#--------------------------

if args_te['plot_vf']:
    assert args_tr['dim_x']==2

    def f(y:np.ndarray):
        Y = torch.tensor(y, dtype=torch.float, device=device)
        X = encoder(Y).view(-1, 2)
        batchsize = X.shape[0]
        T = torch.zeros(batchsize, dtype=torch.float, device=device)
        F = decoder(system(T, X)).detach().numpy()
        return F[:,0], F[:,1]

    m = 20
    fig, ax = plt.subplots(1, 1)
    x1, x2, x1_vel, x2_vel = plot_flow_field(ax, f, plot_range_x, plot_range_y,
        quiver=False, n_grid=m)

    # save figure
    ax.set_aspect('equal', 'box')
    figpath = outprefix+'vf.png'
    plt.ylim(plot_range_x); plt.ylim(plot_range_y)
    plt.savefig(figpath)
    print('save', figpath)

    # save vf values
    vfval = np.concatenate((x1.reshape(m*m,1),
                            x2.reshape(m*m,1),
                            x1_vel.reshape(m*m,1),
                            x2_vel.reshape(m*m,1)), axis=1)
    datpath = outprefix+'vf.txt'
    np.savetxt(datpath, vfval, fmt='%.3f', encoding='utf8', comments='', header='x y u v')
    print('save', datpath)

#--------------------------
# Plot generated trajectory
#--------------------------

if args_te['plot_traj']:
    assert args_tr['dim_x']==2
    lag = max(1, args_tr['initer_lag'])

    # generate
    init_X = encoder(
        torch.FloatTensor(np.array(args_te['geninit']), device=device) ).detach().view(1, 2)
    integration_time = torch.linspace(
        0.0, args_te['gentend'], steps=args_te['genlen'], device=device).detach()
    if 'discrete' in args_tr and args_tr['discrete']:
        raise NotImplementedError()
    else:
        out = decoder(odeint(system, init_X, integration_time,
            rtol=1e-3, atol=1e-3, method='rk4').view(-1, 2)).detach().numpy()
            # (len_episode-lag+1, dim_y)

    fig, ax = plt.subplots(1, 1)
    ax.plot(out[:,0], out[:,1])

    # save
    ax.set_aspect('equal')
    figpath = outprefix+'traj.png'
    plt.ylim(plot_range_x); plt.ylim(plot_range_y)
    plt.savefig(figpath)
    print('save', figpath)

    datpath = outprefix+'traj.txt'
    np.savetxt(datpath, out, fmt='%.6e', encoding='utf8', comments='', header='x y')
    print('save', datpath)

#--------------------------
# Plot Lyapunov function
#--------------------------

if args_te['plot_lyap']:
    assert args_tr['dim_x']==2

    a = args_te['plot_range_x']
    b = args_te['plot_range_y']
    m = 50
    x1, x2 = torch.meshgrid([torch.linspace(-a,a,m), torch.linspace(-b,b,m)])
    x = torch.cat([x1.reshape(m*m,1), x2.reshape(m*m,1)], dim=1)

    if system.transform:
        z_aug = system.phi(x).detach()
        z = z_aug[:, :2]
    else:
        z = x.clone()
    V_x = system.V(z).detach().numpy().reshape(m,m)
    sqrt_V_x = np.sqrt(V_x)

    # contour
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(x1.detach().numpy(), x2.detach().numpy(), sqrt_V_x, levels=10)

    # save contour
    figpath = outprefix+'lyap_contour.png'
    plt.ylim(plot_range_x); plt.ylim(plot_range_y)
    plt.savefig(figpath)
    print('save', figpath)

    # save V values
    Vvalue = np.concatenate((x1.detach().numpy().reshape(m*m,1),
                             x2.detach().numpy().reshape(m*m,1),
                             sqrt_V_x.reshape(m*m,1)), axis=1)
    datpath = outprefix+'lyap_Vvalue.txt'
    np.savetxt(datpath, Vvalue, fmt='%.3f', encoding='utf8', comments='', header='x y z')
    print('save', datpath)

    # V parameters
    datpath = outprefix+'lyap_Vparam.txt'
    with open(datpath, 'w') as fp:
        for name, param in system.V.invset.named_parameters():
            print(name, param)
            fp.write(str(name)+' = '+str(param))
            fp.write('\n')
    print('save', datpath)

#--------------------------
# Test
#--------------------------

if args_te['test']:
    lag = max(1, args_tr['initer_lag'])

    # load test data
    data_te_T, data_te_Y, data_te_dotY = loaddata(args_te['datadir'], args_te['datafile_te'],
        device=device, dotY=True)

    if len(args_te['testepi'])==1 and args_te['testepi'][0]==-1:
        args_te['testepi'] = range(data_te_T.shape[0])

    b = len(args_te['testepi'])
    l = data_te_T[0].shape[0]
    dif_all = np.zeros((b, l))
    for i, idx in enumerate(args_te['testepi']):
        # generate
        T = data_te_T[idx]
        dt = (T[1]-T[0]).item()
        Y = data_te_Y[idx]
        X = encoder(Y)
        init_X = initer(X[:lag]).view(1, args_tr['dim_x']).detach()

        integration_time = torch.linspace(0.0, T[l-1]-T[0], steps=l, device=device).detach()

        if 'discrete' in args_tr and args_tr['discrete']:
            x = init_X
            out = torch.zeros((l, args_tr['dim_x']), device=device)
            for t in range(l):
                out[t] = x.clone()
                dotx = system(integration_time[t], x).detach()*args_te['normdy']
                x = x + dotx*dt
            out = decoder(out).detach().numpy()
        else:
            out = decoder(odeint(system, init_X, integration_time,
                rtol=1e-3, atol=1e-3, method='rk4').view(-1, args_tr['dim_x'])).detach().numpy()

        out[out!=out] = 1e10

        dif = out-Y.detach().numpy()
        difsq = dif*dif
        rmse = np.sqrt(np.mean(difsq))
        print(idx, 'RMSE', rmse)
        dif_all[i] = np.sqrt(np.sum(difsq, axis=1))

        if args_te['testplotfeat']>=0:
            feat = args_te['testplotfeat']
            fig, ax = plt.subplots(1, 1)
            ax.plot(integration_time.detach().numpy(), out[:,feat])
            ax.plot(integration_time.detach().numpy(), Y[:l,feat])
            ax.set_title('RMSE=%f' % rmse)
            plt.show()

        # save prediction (each test episode)
        if args_te['testsavepred']:
            datpath = outprefix+'test_pred_epiidx%d.txt'%idx
            np.savetxt(datpath, out, fmt='%.6e', encoding='utf8')
            print('save', datpath)

    # save dif (over all test episodes)
    difabs_all = np.abs(dif_all)

    if args_te['testsavedif']:
        datpath = outprefix+'test_difabs_all.txt'
        np.savetxt(datpath, difabs_all, fmt='%.6e', encoding='utf8')
        print('save', datpath)

        stat = np.concatenate((np.linspace(1, T.shape[0], T.shape[0]).reshape(-1,1),
                               np.mean(difabs_all, axis=0).reshape(-1,1),
                               np.std(difabs_all, axis=0).reshape(-1,1)), axis=1)
        datpath = outprefix+'test_difabs_stat.txt'
        np.savetxt(datpath, stat, fmt='%.6e', encoding='utf8', comments='', header='x y err')
        print('save', datpath)
