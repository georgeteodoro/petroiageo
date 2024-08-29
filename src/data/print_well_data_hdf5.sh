#!/bin/bash
#SBATCH --nodes=1           #Numero de Nó
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH -J por_gen       #Nome job
#SBATCH --time=1:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós

#module load python/3.9.1
module load /scratch/app/modulos/sequana/current openmpi/gnu/4.0.1_sequana

source /petrobr/parceirosbr/petrobrasiageo/daniel.campos/venv-py3.12/bin/activate

python3 --version
which python3
python3 -m pip list -v

INPUT='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area1/cube/cube_2_train_wells_baseline.h5'
WELLS_IDXS=(5 3 2)
CONFIG='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/petroiageo/src/alg/configs/ANP/v2/area1/exp2/area_1_config.yaml'

echo ${WELLS_IDXS[@]}
python3 print_well_data_hdf5.py --input $INPUT --config $CONFIG --wells_idxs ${WELLS_IDXS[@]}
