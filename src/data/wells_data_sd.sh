#!/bin/bash
#SBATCH --nodes=1           #Numero de Nó
#SBATCH -p ict_cpu       #Fila (partition) a ser utilizada
#SBATCH -J por_gen       #Nome job
#SBATCH --time=1:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive         #Utilização exclusiva dos nós

module load python/3.9.1

source ../../../venv/bin/activate
FEAT_FILE='../../../data/POV/features/area_1/hdf5/pov_area_1_near_seismic_azimuth_.h5'
POR_FILE='../../../data/POV/porosity/merged/pov_area_1_merged_info.csv'
P_COL='density_por'
TARGET_HDF5_FILE='../../../data/POV/porosity/processed/baseline_area_1.h5'

python3 wells_data3_hdf5.py -f $FEAT_FILE -p $POR_FILE -o $TARGET_HDF5_FILE -p_col $P_COL
