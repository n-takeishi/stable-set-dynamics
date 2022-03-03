#!/bin/bash

outdir="./result/"

mkdir -p ${outdir}

if [ $1 = "vf" ]; then
    mode="--plot-range-x 3.5 --plot-range-y 8 --plot-vf"
elif [ $1 = "lyap" ]; then
    mode="--plot-range-x 2.5 --plot-range-y 5 --plot-lyap"
elif [ $1 = "test" ]; then
    mode="--test --testepi -1 --testsavedif"
fi

array_pre=("vanilla" "stable")

for pre in ${array_pre[@]}
do
    python ../test.py --outdir ${outdir} --indir ./out/ \
        --inprefix ${pre} --outprefix ${pre}${outpreadd} ${mode}
done
