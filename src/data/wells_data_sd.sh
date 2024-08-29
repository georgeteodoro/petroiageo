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
FEAT_FILE='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area1/npy_disp/area_1_FAR.npy'
POR_FILE='../../../../dados/ANP/porosity/merged/v3/merged_area_1_NEAR.csv'
P_COL='density_por'
TARGET_HDF5_FILE='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area1/cube/baseline_cube.h5'
CONFIG_FILE='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/petroiageo/src/alg/configs/ANP/v2/exp1/area_1_config.yaml'

python3 wells_data3_hdf5.py -f $FEAT_FILE -p $POR_FILE -o $TARGET_HDF5_FILE -p_col $P_COL -config $CONFIG_FILE
