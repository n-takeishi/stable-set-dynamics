#!/bin/bash

outdir="./result/"

mkdir -p ${outdir}

opts="\
--outdir ${outdir} \
--plot-range-x 2 \
--plot-range-y 2"

python ../test.py ${opts} --indir ./out/ \
    --inprefix stable --outprefix stable --plot-lyap
