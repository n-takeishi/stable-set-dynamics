#!/bin/bash

outdir="./result/"

mkdir -p ${outdir}

opts="\
--outdir ${outdir} \
--plot-range-x 1.2 \
--plot-range-y 1.2 \
"

array_inprefix=("vanilla" "stable")

if [ $1 = "traj" ]; then
    mode="--plot-traj --gentend 20.0 --genlen 200 --geninit -0.1 0.1"

elif [ $1 = "vf" ]; then
    mode="--plot-vf";

elif [ $1 = "test" ]; then
    mode="--test --testsavedif --testepi -1"
fi

for pre in ${array_inprefix[@]}
do
    python ../test.py ${opts} ${mode} --indir ./out/ ${mode} \
        --inprefix ${pre} --outprefix ${pre}${outpreadd}
done
