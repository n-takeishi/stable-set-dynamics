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
--no-transform \
--activation elu \
--invset-type sphere \
--invset-mode surf \
--zero-slack \
--learnrate 1e-4 \
--epoch 20000 \
--weightdecay 1e-4 \
--intv-eval 10 \
--intv-log 100 \
--seed 12345 \
--disp"

mkdir -p ${outdir}

python ../train.py ${opt0} --prefix vanilla \
    --dims-dyn-mlphid-base 64-64 \
    --no-stability --no-invariance

python ../train.py ${opt0} --prefix stable \
    --dims-dyn-mlphid-base 64-64 \
    --dims-dyn-mlphid-lyap-convexfun 16
