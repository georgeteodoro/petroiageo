#!/bin/bash

NUM_NODES=1
N_WORKERS=( 1 2 4 8 16 32 )
N_FEATURES_FILES=1
N_SELECTED_FEATURES=10


echo Local scaling tests (full execution)

for W in ${N_WORKERS[@]}; do
    JOB_NAME="scaling-local-W${W}-f${N_FEATURES_FILES}-fs${N_SELECTED_FEATURES}-it1"

    sbatch --job-name=$JOB_NAME --nodes=$NUM_NODES --time="1:00" \
        base_run.sh $(($W + 1)) $N_FEATURES_FILES $N_SELECTED_FEATURES $JOB_NAME

done
