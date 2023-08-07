#!/usr/bin/sh

CONFIG_FILE='area1_config.yaml'
BASELINE_POR_FILE='/home/daniel/Documentos/Git/petroiageo/data/ANP/processed/area1/porosity_data_baseline.h5'
#This should be the same one on the CONFIG_FILE
TARGET_POR_FILE='/home/daniel/Documentos/Git/petroiageo/data/ANP/processed/area1/porosity_data.h5'

#Create a copy of the baseline
cp $BASELINE_POR_FILE $TARGET_POR_FILE

python3 -u main.py --config  $CONFIG_FILE --nits 5 --nf 1 --nsf 2 --ntf 5
