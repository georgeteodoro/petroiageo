#!/bin/bash

# IMPORTANT: BEFORE RUNNING FOR THE FIRST TIME, BE AWARE TO
# COPY THE BASELINE POROSITY TO THE ORIG_POR_DATA_PATH
# SO OTHER IRRELEVANT RUNS DONT INTERFERE 
set -x

ORIG=$(pwd)

# When using a virtual environment, which is highly recommended
VENV_PATH=../../venv
source $VENV_PATH/bin/activate
python3 -m pip list -v

ORIG_HDF5_FEATS_FOLDER=/home/dan
ORIG_POR_DATA_PATH=/home/daniel/Downloads/petroiageo/anp/area1/cube/baseline_cube_copy.h5
TARGET_TMP_POR_FILE_NAME=/tmp/test/porosity_data.h5

BASE=$(pwd)
# cd /tmp/test

CONFIG_FILE='./configs/ANP/v2/area1/exp1/area_1_config.yaml'
#If there are a lot of features inside the features folder specified in the config file, this option reduces the number of features loaded for the alg.
#This is mainly for testing purposes as for a real run we should consider all features possible.
NUM_FEATS_TO_CONSIDER=0

FILENAME="run_log"
LOG_FILE_NAME=${BASE}/${FILENAME}.mem.log
# clear previous log
rm -f $LOG_FILE_NAME
rm -f $FILENAME

# The iteration starts at 0
IT_START=0
# Number of iterations to run beggining at IT_START
N_ITS=15
# Number of processes/workers per node to be used
N_PROCS=2
NSF=1
# Number of tested features
NTF=1 

# See python3 -u main.py --help for args list 
time vmstat 5 -S M -t -w >> $LOG_FILE_NAME & mpirun -np $N_PROCS --tag-output --bind-to core --map-by node python3 -u main.py --config $CONFIG_FILE --it $IT_START --nits $N_ITS --nsf $NSF --nf $NUM_FEATS_TO_CONSIDER --ntf $NTF --p-dfs --t-shd --no-abort| tee $FILENAME.log
