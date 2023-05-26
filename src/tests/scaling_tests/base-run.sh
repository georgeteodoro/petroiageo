#!/bin/bash
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós
#SBATCH --mail-type=ALL
#SBATCH --mail-user=guns945@gmail.com


set -x

ORIG=$(pwd)

module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana
source /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/venv/bin/activate

N_PROCS=$1
N_FEATURES_FILES=$2
N_SELECTED_FEATURES=$3
EXEC_NAME=$4

# Should run on local storage (SSD) instead of distributed storage
ON_LOCAL=1
if [ $ON_LOCAL -eq 1 ] 
    then
    srun -N${SLURM_JOB_NUM_NODES} rm -rf /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} cp config.yaml /tmp/test/
    srun -N${SLURM_JOB_NUM_NODES} cp ../../alg/*.py /tmp/test
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test/data
    srun -N${SLURM_JOB_NUM_NODES} mkdir /tmp/test/data/h5_features
    srun -N${SLURM_JOB_NUM_NODES} \
        cp ../../../data/POV/features/h5_features/*.h5 \
        /tmp/test/data/h5_features
    srun -N${SLURM_JOB_NUM_NODES} \
        cp ../../../data/POV/porosity_data_bckup-434x323x251.h5 \
        /tmp/test/data/porosity_data.h5
    srun -N${SLURM_JOB_NUM_NODES} cd /tmp/test
else
    echo "================NOT RUNNING ON LOCAL!!!!!!!!!"
fi
cd /tmp/test
rm cur-r*

pwd
ls

# Memory and CPU usage global profiling
vmstat 5 -S M -t -w > ${ORIG}/${EXEC_NAME}.mem.log &

# Actual execution
/usr/bin/time -v mpirun -np $N_PROCS --map-by node --tag-output --bind-to core \
    --oversubscribe --output-filename ${ORIG}/${EXEC_NAME} \
    --merge-stderr-to-stdout perf stat python3 -u main.py --no-wp --nits 1 \
    --nf $N_FEATURES_FILES --nsf $N_SELECTED_FEATURES --config config.yaml \
    | tee $EXEC_NAME.log

# /bin/time -v mpirun -np 3 --bind-to core --output-filename logg --merge-stderr-to-stdout perf stat python3 main.py --config config.yaml --nf 1 --nsf 10 --ntf 10 --nits 1

if [ $ON_LOCAL -eq 1 ]; then
    cp $FILENAME.log $ORIG
    ls
fi
