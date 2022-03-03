#!/usr/bin/bash

outdir="./ibpm_result/"

opt="-Re 100 -nx 450 -ny 200 -ngrid 4 -length 9 -xoffset -1 -yoffset -2 -xshift 0.75 -outdir ${outdir}"

./ibpm/build/ibpm ${opt} -nsteps 500 -restart 500 -tecplot 500 -geom ./cylinder2Pa.geom -dt .02 -ubf 0 -tecplotallgrids 1
./ibpm/build/ibpm -ic ${outdir}/ibpm00500.bin ${opt} -nsteps 50 -restart 50 -tecplot 50 -geom cylinder2PaPlunge.geom -dt .002 -ubf 1 -tecplotallgrids 1
./ibpm/build/ibpm -ic ${outdir}/ibpm00550.bin ${opt} -nsteps 5000 -restart 500 -tecplot 500 -geom cylinder2Pa.geom -dt .02 -ubf 0 -tecplotallgrids 0
./ibpm/build/ibpm -ic ${outdir}/ibpm05500.bin ${opt} -nsteps 10000 -restart 5000 -tecplot 10 -geom cylinder2Pa.geom -dt .02 -ubf 0 -tecplotallgrids 0
./ibpm/build/ibpm -ic ${outdir}/ibpm15000.bin ${opt} -nsteps 25500 -restart 5000 -tecplot 10 -geom cylinder2Pa.geom -dt .02 -ubf 0 -tecplotallgrids 0
