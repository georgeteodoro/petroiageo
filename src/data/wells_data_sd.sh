#!/bin/bash
#SBATCH --nodes=1           #Numero de Nó
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH -J por_gen       #Nome job
#SBATCH --time=1:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós

#module load python/3.9.6

source ../../../meu_python/bin/activate
FEAT_FILE=../../data/POV/features/NEAR.npy
POR_FILE=../../data/POV/raw/porosity-canal.txt
TARGET_HDF5_FILE=../../data/POV/processed/porosity_data.h5

python3 wells_data3_hdf5.py --base-feat-file  $FEAT_FILE --porosity-file-path  $POR_FILE --hdf5-file-path $TARGET_HDF5_FILE
