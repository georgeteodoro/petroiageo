#!/bin/bash
#SBATCH --nodes=1           #Numero de Nó
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH -J por_gen       #Nome job
#SBATCH --time=1:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós

module load python/3.9.6

source ../../../../venv/bin/activate
FEATS_FOLDER='../../../../data/POV/features/area_1/'
TARGET_FOLDER='../../../../data/POV/features/area_1/hdf5'
CONFIG_FILE='../../alg/pov_area_1_config.yaml'

python3 seismic_data3_hdf5.py -f $FEATS_FOLDER -o $TARGET_FOLDER -config $CONFIG_FILE
