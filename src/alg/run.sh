#!/bin/bash
#SBATCH --nodes=1		#Numero de Nós
#SBATCH -p ict_cpu		#Fila (partition) a ser utilizada
#SBATCH -J area_1_job		#Nome job
#SBATCH --time=24:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive		#Utilização exclusiva dos nós
#SBATCH --no-requeue		# Não ressubmete um job se ele der erro

set -x

ORIG=$(pwd)

module load python/3.9.6
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
#source /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/venv/bin/activate

pip3 list

ON_LOCAL=1

if [ $ON_LOCAL -eq 1 ] 
    then
    srun -N${SLURM_JOB_NUM_NODES} rm -rf /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} cp * /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test/features
    srun -N${SLURM_JOB_NUM_NODES} cp -r ../../data/ANP/features/area1/hdf5/*.h5 /tmp/test/features/
    srun -N${SLURM_JOB_NUM_NODES} cp ../../data/ANP/processed/area_1/porosity_data.h5 /tmp/test/porosity_data.h5
    srun -N${SLURM_JOB_NUM_NODES} cd /tmp/test
#    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test/tmp_data
    ls
else
    echo "================NOT RUNNING ON LOCAL!!!!!!!!!"
fi

BASE=$(pwd)
cd /tmp/test

W=16    # number of local workers
FILENAME="teste_new_sampling_area1"

CONFIG_FILE='./area1_config.yaml'
#If there are a lot of features inside the features folder specified in the config file, this option reduces the number of features loaded for the alg.
#This is mainly for testing purposes as for a real run we should consider all features possible.
NUM_FEATS_TO_CONSIDER=4

#There are some other params that you could configure
# see python3 main.py --help for more

vmstat 5 -S M -t -w > ${BASE}/${FILENAME}.mem.log &
mpirun -np $(($SLURM_JOB_NUM_NODES * $W)) --map-by node --tag-output --bind-to core --oversubscribe python3 -u main.py --config $CONFIG_FILE --no-wp --local | tee $FILENAME.log

