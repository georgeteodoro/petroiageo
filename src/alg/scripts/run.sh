#!/bin/bash
#SBATCH --nodes=1		#Numero de Nós
#SBATCH -p ict_cpu		#Fila (partition) a ser utilizada
#SBATCH -J area_1_job		#Nome job
#SBATCH --time=24:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive		#Utilização exclusiva dos nós
#SBATCH --no-requeue		# Não ressubmete um job se ele der erro

# IMPORTANT: BEFORE RUNNING FOR THE FIRST TIME, BE AWARE TO
# COPY THE BASELINE POROSITY TO THE ORIG_POR_DATA_PATH
# SO OTHER IRRELEVANT RUNS DONT INTERFERE 
set -x

ORIG=$(pwd)

module load python/3.9.6
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
#source /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/venv/bin/activate

pip3 list

ON_LOCAL=1

ORIG_HDF5_FEATS_FOLDER=../../data/ANP/features/area1/hdf5/
ORIG_POR_DATA_PATH=../../data/ANP/processed/area_1/porosity_data.h5
TARGET_TMP_POR_FILE_NAME=/tmp/test/porosity_data.h5

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
NUM_FEATS_TO_CONSIDER=4

FILENAME="teste_new_sampling_area1"
LOG_FILE_NAME=${BASE}/${FILENAME}.mem.log
# clear previous log
rm -f $LOG_FILE_NAME
rm -f $FILENAME

#There are some other params that you could configure
# see python3 main.py --help for more
# The its start at 1 NOT 0.
IT_START=1
IT_END=22

W=16 # number of local workers

for it in $(seq $IT_START $IT_END)
do

    vmstat 5 -S M -t -w >> $LOG_FILE_NAME &
    mpirun -np $(($SLURM_JOB_NUM_NODES * $W)) --map-by node --tag-output --bind-to core --oversubscribe python3 -u main.py --it $it --config $CONFIG_FILE --no-wp --local | tee $FILENAME.log

    if [ $ON_LOCAL -eq 1 ]
        then
        #Get out of tmp/test
        cd ../..
        # Save curr it results into the original por file
        cp $TARGET_TMP_POR_FILE_NAME $ORIG_POR_DATA_PATH
        # Get back to the right folder
        cd /tmp/test
    else
        echo "DONT NEED TO COPY DATA! NOT RUNNING ON LOCAL!!!!!!!!"
    fi
done