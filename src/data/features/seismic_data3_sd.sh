#!/bin/bash
#SBATCH --nodes=1		#Numero de Nós
#SBATCH -p ict_cpu		#Fila (partition) a ser utilizada
#SBATCH -J por_gen		#Nome job
#SBATCH --time=1:00:00		#Tempo máximo de execução
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive		#Utilização exclusiva dos nós

module load python/3.9.1

source ../../../../venv-py3.9.1/bin/activate
FEATS_FOLDER='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area2/npy'
TARGET_FOLDER='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/data/ANP/v2/area2/npy_disp/'
CONFIG_FILE='/petrobr/parceirosbr/petrobrasiageo/daniel.campos/petroiageo/src/alg/configs/ANP/v2/area_2_config.yaml'

python3 seismic_data3_npy.py -f $FEATS_FOLDER -o $TARGET_FOLDER --config $CONFIG_FILE
