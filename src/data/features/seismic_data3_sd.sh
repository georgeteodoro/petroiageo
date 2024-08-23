#!/bin/bash
#SBATCH --nodes=1		#Numero de Nós
#SBATCH -p ict_cpu		#Fila (partition) a ser utilizada
#SBATCH -J por_gen		#Nome job
#SBATCH --time=1:00:00		#Tempo máximo de execução
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive		#Utilização exclusiva dos nós

module load python/3.9.6

source ../../../../venv/bin/activate
FEATS_FOLDER='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area1/npy'
TARGET_FOLDER='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area1/hdf5'
CONFIG_FILE='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/petroiageo/src/alg/configs/ANP/v2/area_1_config.yaml'

python3 seismic_data3_hdf5.py -f $FEATS_FOLDER -o $TARGET_FOLDER --config $CONFIG_FILE
