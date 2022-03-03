#!/bin/bash

outdir="./out/"
datadir="./"
dataname="data"

opt0="\
--train-single \
--discrete \
--datadir ${datadir} \
--datafile-tr ${dataname}_tr.mat \
--datafile-va ${dataname}_va.mat \
--outdir ${outdir} \
--activation elu \
--invset-type sphere \
--invset-mode surf \
--zero-slack \
--dim-aug-tran 1 \
--learnrate 1e-4 \
--epoch 5000 \
--tol 1e-4 \
--weightdecay 1e-4 \
--intv-eval 10 \
--intv-log 100 \
--disp"

mkdir -p ${outdir}

python ../train.py ${opt0} --prefix vanilla \
    --seed 1 --dims-dyn-mlphid-base 128-128 --dims-dyn-mlphid-tran 32 \
    --no-stability --no-invariance

python ../train.py ${opt0} --prefix staeq \
    --seed 1 --dims-dyn-mlphid-base 128-128 --dims-dyn-mlphid-tran 32 \
    --dims-dyn-mlphid-lyap-convexfun 128 --dim-invset 2 --invset-sphere-init 0

python ../train.py ${opt0} --prefix stainv \
    --seed 1 --dims-dyn-mlphid-base 128-128 --dims-dyn-mlphid-tran 32 \
    --dims-dyn-mlphid-lyap-convexfun 128 --dim-invset 2 --invset-sphere-init 1
