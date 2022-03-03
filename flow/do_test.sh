#!/bin/bash

outdir="./result/"

mkdir -p ${outdir}

opts="\
--outdir ${outdir} \
--datafile-te data_te.mat"

mode="--test --testsavepred --testplotfeat 0 --normdy 0.1 --testepi 0"

pre="vanilla"
python ../test.py ${opts} ${mode} --indir ./out/ --inprefix ${pre} --outprefix ${pre}

pre="staeq"
python ../test.py ${opts} ${mode} --indir ./out/ --inprefix ${pre} --outprefix ${pre}

pre="stainv"
python ../test.py ${opts} ${mode} --indir ./out/ --inprefix ${pre} --outprefix ${pre}
