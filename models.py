import math
import numpy as np
import torch

from stadyn.mlp import MLP
from stadyn.initializer import Initializer
from stadyn.dynamics import Dynamics


def new(device:torch.device, args:dict):
    dim_x = args['dim_x']
    dim_y = args['dim_y']

    # dynamics model
    np.random.seed(args['seed']+1); torch.manual_seed(args['seed']+1)

    if 'dim_invset' not in args:
        args['dim_invset'] = dim_x

    if args['dims_dyn_mlphid_tran'] is None:
        args['dims_dyn_mlphid_tran'] = [dim_x*2, dim_x*2]
    if args['dims_dyn_mlphid_base'] is None:
        args['dims_dyn_mlphid_base'] = [dim_x*2, dim_x*4, dim_x*2]
    if args['dims_dyn_mlphid_lyap_warping'] is None:
        args['dims_dyn_mlphid_lyap_warping'] = [dim_x*2, dim_x*2]
    if args['dims_dyn_mlphid_lyap_convexfun'] is None:
        args['dims_dyn_mlphid_lyap_convexfun'] = [dim_x*2, dim_x*2]

    system = Dynamics(dim_x,
        dim_invset = args['dim_invset'],
        invset_type = args['invset_type'],
        invset_mode = args['invset_mode'],
        invset_learnable = args['invset_learnable'],
        invset_sphere_init = args['invset_sphere_init'],
        transform = not args['no_transform'],
        invariance = not args['no_invariance'],
        stability = not args['no_stability'],
        alpha = args['alpha'],
        dims_hidden_tran = args['dims_dyn_mlphid_tran'],
        dims_hidden_base = args['dims_dyn_mlphid_base'],
        dims_hidden_lyap_warping = args['dims_dyn_mlphid_lyap_warping'],
        dims_hidden_lyap_convexfun = args['dims_dyn_mlphid_lyap_convexfun'],
        dim_aug_tran = args['dim_aug_tran'],
        dim_aug_lyap_warping = args['dim_aug_lyap_warping'],
        activation = args['activation'],
        batchnorm = not args['no_batchnorm'],
        smooth_step = args['smooth_step'],
        zero_slack = args['zero_slack'],
        warping = args['warping'],
        eps = args['eps'],
        dropout=-1).to(device)

    # initializer model
    np.random.seed(args['seed']+3); torch.manual_seed(args['seed']+3)

    if args['initer_lag']>0:
        if args['dims_initer_mlphid'] is None:
            d1 = (2.0*dim_x*args['initer_lag'] + dim_x) / 3.0
            d2 = (dim_x*args['initer_lag'] + 2.0*dim_x) / 3.0
            args['dims_initer_mlphid'] = [math.ceil(d1,), math.ceil(d2)]

        initer = Initializer(dim_x, args['initer_lag'], args['dims_initer_mlphid'],
            activation = args['activation'],
            batchnorm = False).to(device)
    else:
        initer = torch.nn.Sequential().to(device)

    encoder = torch.nn.Sequential()
    decoder = torch.nn.Sequential()

    # return
    return system, encoder, decoder, initer


def load(modeldir:str='',
         modelfile_system:str='', modelfile_encoder:str='', modelfile_decoder:str='',
         modelfile_tran:str='', modelfile_initer:str='',
         system:torch.nn.Module=None, encoder:torch.nn.Module=None,
         decoder:torch.nn.Module=None, initer:torch.nn.Module=None, disp:bool=False):

    if modelfile_system and system is not None:
        modelpath = modeldir+'/'+modelfile_system
        system.load_state_dict(torch.load(modelpath))
        if disp:
            print('load system from', modelpath)

    if modelfile_encoder and encoder is not None:
        modelpath = modeldir+'/'+modelfile_encoder
        encoder.load_state_dict(torch.load(modelpath))
        if disp:
            print('load encoder from', modelpath)

    if modelfile_decoder and decoder is not None:
        modelpath = modeldir+'/'+modelfile_decoder
        decoder.load_state_dict(torch.load(modelpath))
        if disp:
            print('load decoder from', modelpath)

    if modelfile_tran and system is not None:
        modelpath = modeldir+'/'+modelfile_tran
        system.phi.load_state_dict(torch.load(modelpath))
        if disp:
            print('load transform from', modelpath)

    if modelfile_initer and initer is not None:
        modelpath = modeldir+'/'+modelfile_initer
        initer.load_state_dict(torch.load(modelpath))
        if disp:
            print('load initer from', modelpath)
