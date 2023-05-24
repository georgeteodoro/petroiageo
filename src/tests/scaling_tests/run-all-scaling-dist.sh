#!/bin/bash

N_NODES=( 1 2 4 8 16 32 )
N_WORKERS=( 32 )
N_FEATURES_FILES=1
N_SELECTED_FEATURES=10


echo Distributed scaling tests (full execution)

for NP in ${N_NODES[@]}; do
    JOB_NAME="scaling-dist-np${NP}-w${N_WORKERS}-f${N_FEATURES_FILES}-fs${N_SELECTED_FEATURES}-it1"

    sbatch --job-name=$JOB_NAME --nodes=$NP --time="1:00" \
        base_run.sh $(($NP * $N_WORKERS + 1)) $N_FEATURES_FILES \
        $N_SELECTED_FEATURES $JOB_NAME

done
