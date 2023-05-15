#!/bin/bash
#SBATCH --nodes=1           #Numero de Nó
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH -J por_gen       #Nome job
#SBATCH --time=1:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós

module load python/3.9.6

source ../../../../meu_python/bin/activate
FEATS_FOLDER='../../../data/POV/features/'
TARGET_FOLDER='../../../data/POV/features/h5_features/'

python3 seismic_data3_hdf5.py --base-feat-folder $FEATS_FOLDER --target-h5-folder $TARGET_FOLDER
