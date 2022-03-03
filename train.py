import os, json, argparse, time, math
import numpy as np
import torch
from torchdiffeq import odeint, odeint_adjoint

from stadyn.utils import *
import models

#--------------------------
# Configuration
#--------------------------

parser = argparse.ArgumentParser(description='stable dynamics learning')

# input
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--datafile-tr', type=str, required=True)
parser.add_argument('--datafile-va', type=str, required=True)
# output
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--prefix', type=str, required=True)
# training mode
parser.add_argument('--train-single', action='store_true', default=False)
parser.add_argument('--train-multi', action='store_true', default=False)
# model (common)
parser.add_argument('--discrete', action='store_true', default=False)
parser.add_argument('--activation', type=str,
    choices=['relu','leakyrelu','elu','softplus','prelu'], default='elu')
parser.add_argument('--no-batchnorm', action='store_true', default=False)
# model (dynamics)
parser.add_argument('--no-transform', action='store_true', default=False)
parser.add_argument('--no-invariance', action='store_true', default=False)
parser.add_argument('--no-stability', action='store_true', default=False)
parser.add_argument('--warping', action='store_true', default=False)
parser.add_argument('--dim-invset', type=int, default=None)
parser.add_argument('--invset-type', type=str,
    choices=['sphere','quadric','linear'], required=True)
parser.add_argument('--invset-mode', type=str,
    choices=['vol','surf'], required=True)
parser.add_argument('--invset-learnable', action='store_true', default=False)
parser.add_argument('--invset-sphere-init', type=float, default=1.0)
parser.add_argument('--dims-dyn-mlphid-tran', type=str, default=None)
parser.add_argument('--dims-dyn-mlphid-base', type=str, default=None)
parser.add_argument('--dims-dyn-mlphid-lyap-convexfun', type=str, default=None)
parser.add_argument('--dims-dyn-mlphid-lyap-warping', type=str, default=None)
parser.add_argument('--dim-aug-tran', type=int, default=0)
parser.add_argument('--dim-aug-lyap-warping', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--zero-slack', action='store_true', default=False)
parser.add_argument('--smooth-step', type=float, default=-1.0)
# model (initializer)
parser.add_argument('--initer-lag', type=int, default=0)
parser.add_argument('--dims-initer-mlphid', type=str, default=None)
# optimization
parser.add_argument('--batchsize', type=int, default=99999)
parser.add_argument('--batchmode', type=str,
    choices=['epi','obs'], default='epi')
parser.add_argument('--learnrate', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--weightdecay', type=float, default=1e-4)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--gradclip', type=float, default=-1.0)
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--reg-lyapvalue', type=float, default=-1.0)
parser.add_argument('--reg-lyapvalue-mode', type=str,
    choices=['mean','max'], default='mean')

# others
parser.add_argument('--disp', action='store_true', default=False)
parser.add_argument('--intv-eval', type=int, default=10)
parser.add_argument('--intv-log', type=int, default=10)
parser.add_argument('--seed', type=int, default=1234567890)
parser.add_argument('--no-cuda', action='store_true', default=False)

args = vars(parser.parse_args())

outprefix = args['outdir']+'/'+args['prefix']+'_'

for key in args:
    if key.startswith('dims_') and args[key] is not None:
        args[key] = [int(s.strip()) for s in args[key].split('-')]

# set device
device = torch.device('cuda' if not args['no_cuda'] and torch.cuda.is_available() else 'cpu')
args['device'] = str(device)
if args['disp']:
    print('chosen device:', device)

#--------------------------
# Data
#--------------------------

np.random.seed(args['seed']); torch.manual_seed(args['seed'])

data_tr_T, data_tr_Y, data_tr_dotY = loaddata(
    args['datadir'], args['datafile_tr'], device=device, dotY=True)
data_va_T, data_va_Y, data_va_dotY = loaddata(
    args['datadir'], args['datafile_va'], device=device, dotY=True)

num_episodes_tr = data_tr_Y.shape[0]
num_episodes_va = data_va_Y.shape[0]

dim_y = data_tr_Y.shape[2]
dim_x = dim_y
args['dim_x'] = dim_x
args['dim_y'] = dim_y

if args['dim_invset'] is None:
    args['dim_invset'] = dim_x

data_tr_T_flat = data_tr_T.reshape(-1,1)
data_tr_Y_flat = data_tr_Y.reshape(-1,dim_y)
data_tr_dotY_flat = data_tr_dotY.reshape(-1,dim_y)

data_va_T_flat = data_va_T.reshape(-1,1)
data_va_Y_flat = data_va_Y.reshape(-1,dim_y)
data_va_dotY_flat = data_va_dotY.reshape(-1,dim_y)

#--------------------------
# Models
#--------------------------

system, encoder, decoder, initer = models.new(device, args)

#--------------------------
# Dynamics training (single-step)
#--------------------------

np.random.seed(args['seed']+5); torch.manual_seed(args['seed']+5)

if args['train_single']:
    args['suffix'] = 'single'
    if args['disp']:
        print()
        print('start training dynamics (single-step)')
        print()

    # prepare optimizer
    params = list(system.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params,
        lr=args['learnrate'], weight_decay=args['weightdecay'], amsgrad=args['amsgrad'])

    # define training procedures
    def train():
        system.train(); encoder.train(); decoder.train()

        total_loss = 0.0
        total_dur = 0.0

        if args['batchmode']=='epi':
            idx_all = np.random.permutation(num_episodes_tr)
        elif args['batchmode']=='obs':
            idx_all = np.random.permutation(data_tr_Y_flat.shape[0])

        k = 0
        while k*args['batchsize'] < idx_all.shape[0]:
            start = time.time()
            optimizer.zero_grad()

            idx = idx_all[k*args['batchsize'] : min((k+1)*args['batchsize'], idx_all.shape[0])]
            if args['batchmode']=='epi':
                T = data_tr_T[idx].reshape(-1,1)
                Y = data_tr_Y[idx].reshape(-1,dim_y)
                dotY = data_tr_dotY[idx].reshape(-1,dim_y)
            elif args['batchmode']=='obs':
                T = data_tr_T_flat[idx].view(-1,1)
                Y = data_tr_Y_flat[idx].view(-1,dim_y)
                dotY = data_tr_dotY_flat[idx].view(-1,dim_y)

            dotX, Vstat = system(T, encoder(Y), return_V=args['reg_lyapvalue_mode'])
            out = decoder(dotX)
            loss = torch.nn.functional.mse_loss(out, dotY, reduction='mean')

            if args['reg_lyapvalue']>0.0:
                loss_ = loss + args['reg_lyapvalue']*Vstat
            else:
                loss_ = loss

            loss_.backward()
            if args['gradclip']>0.0:
                torch.nn.utils.clip_grad_value_(params, args['gradclip'])
            optimizer.step()

            total_dur += time.time() - start
            total_loss += loss.item()
            k += 1
        total_loss /= k

        return total_loss, total_dur

    # define evaluation procedures
    def eval():
        system.eval(); encoder.eval(); decoder.eval()

        out = decoder(system(data_va_T_flat, encoder(data_va_Y_flat)))
        loss = torch.nn.functional.mse_loss(out, data_va_dotY_flat, reduction='mean').item()

        return loss

    # make output directory
    os.makedirs(args['outdir'], exist_ok=True)

    # save args
    with open(outprefix+'args_%s.json'%args['suffix'], 'w') as fp:
        json.dump(args, fp, sort_keys=True, indent=4)

    # reset log file
    with open(outprefix+'log_%s.txt'%args['suffix'], 'w', encoding='utf-8') as fp:
        fp.write('')

    # main loop
    loss_va = 1e100
    loss_va_best = 1e99
    comment = ''
    for epoch in range(args['epoch']):
        loss_tr, dur_tr = train()

        if epoch % args['intv_eval'] == 0:
            loss_va = eval()

            if loss_va < loss_va_best:
                loss_va_best = loss_va

                torch.save(system.state_dict(), outprefix+'system_%s.pt'%args['suffix'])
                if 'save system' not in comment:
                    comment += 'save system  '

        if epoch % args['intv_log'] == 0:
            log = '%06d  tr=%0.2e  va=%0.2e  va_best=%0.2e  dur=%0.3fs/epoch  %s' % (epoch,
                loss_tr, loss_va, loss_va_best, dur_tr, comment)
            if args['disp']:
                print(log)
            with open(outprefix+'log_%s.txt'%args['suffix'], 'a', encoding='utf-8') as fp:
                fp.write(log+'\n')
            comment = ''

        if loss_tr < args['tol']:
            break

    # reload best model
    system.load_state_dict(torch.load(outprefix+'system_%s.pt'%args['suffix']))

#--------------------------
# Dynamics training (multi-step)
#--------------------------

METHOD = 'rk4' #'dopri5'
ACTIVATION = 'softplus'
MAX_NUM_STEPS = 1000
RTOL = 1e-3
ATOL = 1e-3
TIME_DEPENDENT = False
if METHOD=='dopri5':
    OPTIONS = {'max_num_steps':MAX_NUM_STEPS}
else:
    OPTIONS = {}

np.random.seed(args['seed']+7); torch.manual_seed(args['seed']+7)

if args['train_multi']:
    args['suffix'] = 'multi'
    if args['disp']:
        print()
        print('start training dynamics (multi-step)')
        print()

    assert args['batchmode'] == 'epi'
    assert not args['discrete']

    lag = max(1, args['initer_lag'])

    # prepare optimizer
    params = list(system.parameters()) \
        + list(encoder.parameters()) + list(decoder.parameters()) + list(initer.parameters())
    optimizer = torch.optim.Adam(params,
        lr=args['learnrate'], weight_decay=args['weightdecay'], amsgrad=args['amsgrad'])

    # define training procedures
    def train():
        system.train(); encoder.train(); decoder.train(); initer.train()

        # set (only) batch normalization to eval mode
        system.apply(set_bn_eval)
        encoder.apply(set_bn_eval)
        decoder.apply(set_bn_eval)
        initer.apply(set_bn_eval)

        total_loss = 0.0
        total_dur = 0.0
        k = 0
        idx_all = np.random.permutation(num_episodes_tr)
        while k*args['batchsize'] < num_episodes_tr:
            start = time.time()
            optimizer.zero_grad()

            idx = idx_all[k*args['batchsize'] : min((k+1)*args['batchsize'], num_episodes_tr)]

            loss = 0.0
            for i in idx:
                T = data_tr_T[i]
                Y = data_tr_Y[i]
                X = encoder(Y)
                init_X = initer(X[:lag]).view(1, dim_x)
                integration_time = T[lag-1:] - T[lag-1]

                out = decoder(odeint(system, init_X, integration_time,
                              rtol=RTOL, atol=ATOL, method=METHOD, options=OPTIONS).view(-1, dim_x))
                              # (len_episode-lag+1, dim_y)
                loss += torch.nn.functional.mse_loss(out, Y[lag-1:], reduction='mean')
            loss = loss / len(idx)
            loss.backward()
            if args['gradclip']>0.0:
                torch.nn.utils.clip_grad_value_(params, args['gradclip'])
            optimizer.step()

            total_dur += time.time() - start
            total_loss += loss.item()
            k += 1

        total_loss /= k
        return total_loss, total_dur

    # define evaluation procedures
    def eval():
        system.eval(); encoder.eval(); decoder.eval(); initer.eval()

        loss = 0.0
        for i in range(num_episodes_va):
            T = data_va_T[i]
            Y = data_va_Y[i]
            X = encoder(Y)
            init_X = initer(X[:lag]).view(1, dim_x)
            integration_time = T[lag-1:] - T[lag-1]

            out = decoder(odeint(system, init_X, integration_time,
                         rtol=RTOL, atol=ATOL, method=METHOD, options=OPTIONS).view(-1, dim_x))
            loss += torch.nn.functional.mse_loss(out, Y[lag-1:], reduction='mean').item()
        loss = loss / num_episodes_va

        return loss

    # make output directory
    os.makedirs(args['outdir'], exist_ok=True)

    # save args
    with open(outprefix+'args_%s.json'%args['suffix'], 'w') as fp:
        json.dump(args, fp, sort_keys=True, indent=4)

    # reset log file
    with open(outprefix+'log_%s.txt'%args['suffix'], 'w', encoding='utf-8') as fp:
        fp.write('')

    # main loop
    loss_va = 1e100
    loss_va_best = 1e99
    comment = ''
    for epoch in range(args['epoch']):
        loss_tr, dur_tr = train()

        if epoch % args['intv_eval'] == 0:
            loss_va = eval()

            if loss_va < loss_va_best:
                loss_va_best = loss_va

                if args['initer_lag']>0:
                    torch.save(initer.state_dict(), outprefix+'initer_%s.pt'%args['suffix'])
                    if 'save initer' not in comment:
                        comment += 'save initer  '
                torch.save(system.state_dict(), outprefix+'system_%s.pt'%args['suffix'])
                if 'save system' not in comment:
                    comment += 'save system  '

        if epoch % args['intv_log'] == 0:
            log = '%06d  tr=%0.2e  va=%0.2e  va_best=%0.2e  dur=%0.3fs/epoch  %s' % (epoch,
                loss_tr, loss_va, loss_va_best, dur_tr, comment)
            if args['disp']:
                print(log)
            with open(outprefix+'log_%s.txt'%args['suffix'], 'a', encoding='utf-8') as fp:
                fp.write(log+'\n')
            comment = ''

        if loss_tr < args['tol']:
            break

