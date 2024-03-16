#!/bin/bash
#SBATCH --nodes=8		    #Numero de Nós
#SBATCH -p ict_cpu		    #Fila (partition) a ser utilizada
#SBATCH -J area_1_job	    #Nome job
#SBATCH --time=4:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive		    #Utilização exclusiva dos nós
#SBATCH --no-requeue	    # Não ressubmete um job se ele der erro

# IMPORTANT: BEFORE RUNNING FOR THE FIRST TIME, BE AWARE TO
# COPY THE BASELINE POROSITY TO THE ORIG_POR_DATA_PATH
# SO OTHER IRRELEVANT RUNS DONT INTERFERE 
set -x

ORIG=$(pwd)

# module load python/3.9.6
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
module load gcc/6.5_sequana

# When using a virtual environment, which is highly recommended
VENV_PATH=../../../venv
source $VENV_PATH/bin/activate
python3 -m pip list -v

ORIG_HDF5_FEATS_FOLDER=/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/area1/hdf5/
ORIG_POR_DATA_PATH=/petrobr/parceirosbr/petrobrasiageo/daniel.campos/petroiageo/data/ANP/processed/area_1/porosity_data.h5
TARGET_TMP_POR_FILE_NAME=/tmp/test/porosity_data.h5

ON_LOCAL=0

if [ $ON_LOCAL -eq 1 ] 
    then
    srun -N${SLURM_JOB_NUM_NODES} rm -rf /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} cp * /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test/features
    srun -N${SLURM_JOB_NUM_NODES} cp -r $ORIG_HDF5_FEATS_FOLDER*.h5 /tmp/test/features/
    srun -N${SLURM_JOB_NUM_NODES} cp $ORIG_POR_DATA_PATH $TARGET_TMP_POR_FILE_NAME
    srun -N${SLURM_JOB_NUM_NODES} cd /tmp/test
#    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test/tmp_data
    ls
else
    echo "================NOT RUNNING ON LOCAL!!!!!!!!!"
fi

BASE=$(pwd)
cd /tmp/test

CONFIG_FILE='./area1_config.yaml'
#If there are a lot of features inside the features folder specified in the config file, this option reduces the number of features loaded for the alg.
#This is mainly for testing purposes as for a real run we should consider all features possible.
# NUM_FEATS_TO_CONSIDER=4

FILENAME="run_log"
LOG_FILE_NAME=${BASE}/${FILENAME}.mem.log
# clear previous log
rm -f $LOG_FILE_NAME
rm -f $FILENAME

# The iteration starts at 1
IT_START=1
# Number of iterations to run beggining at IT_START
N_ITS=1
# The window used to consider on top of base features
WINDOW_USED=3
# Number of processes/workers per node to be used
N_PROCS=48

# See python3 -u main.py --help for args list 
time vmstat 5 -S M -t -w >> $LOG_FILE_NAME & mpirun -np $(($SLURM_JOB_NUM_NODES * $N_PROCS)) --map-by node --tag-output --bind-to core --oversubscribe python3 -u main.py --config $CONFIG_FILE --it $IT_START --nits $N_ITS -w $WINDOW_USED| tee $FILENAME.log

if [ $ON_LOCAL -eq 1 ]
    then
    #Get out of tmp/test
    cd ../..
    # Save curr it results into the original por file
    cp $TARGET_TMP_POR_FILE_NAME $ORIG_POR_DATA_PATH
else
    echo "DONT NEED TO COPY DATA! NOT RUNNING ON LOCAL!!!!!!!!"
fi
