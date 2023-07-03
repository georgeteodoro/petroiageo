#!/bin/bash
#SBATCH --nodes=1           #Numero de Nó
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH -J por_gen       #Nome job
#SBATCH --time=1:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós

module load python/3.9.6

source ../../../meu_python/bin/activate
FEAT_FILE='../../data/ANP/features/area1/hdf5/Franco_florin_buzios_28_09-2_azimuth_area1_.h5'
POR_FILE='../../data/ANP/processed/area_1_porsty.npy'
TARGET_HDF5_FILE='../../data/ANP/processed/area_1/porosity_data.h5'

python3 wells_data3_hdf5.py -f  $FEAT_FILE -p  $POR_FILE -o $TARGET_HDF5_FILE
