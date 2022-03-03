#!/bin/bash

outdir="./out/"
datadir="./"
dataname="data"

opt0="\
--train-single \
--datadir ${datadir} \
--datafile-tr ${dataname}_tr.mat \
--datafile-va ${dataname}_va.mat \
--outdir ${outdir} \
--activation elu \
--invset-type sphere \
--invset-mode surf \
--invset-sphere-init 2.0 \
--zero-slack \
--dim-aug-tran 1 \
--learnrate 1e-3 \
--epoch 200000 \
--tol 1e-9 \
--weightdecay 1e-4 \
--intv-eval 2 \
--intv-log 100 \
--seed 12345 \
--smooth-step 0.1 \
--disp"

mkdir -p ${outdir}

python ../train.py ${opt0} --prefix vanilla \
    --dims-dyn-mlphid-base 32-32 --dims-dyn-mlphid-tran 64-64 \
    --no-stability --no-invariance

python ../train.py ${opt0} --prefix stable \
    --dims-dyn-mlphid-base 32-32 --dims-dyn-mlphid-tran 64-64 \
    --dims-dyn-mlphid-lyap-convexfun 128
