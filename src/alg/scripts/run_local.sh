#!/usr/bin/sh
# Esse script executa do forma local sem o MPI o algoritmo. Após definir variáveis
# intuitivamente, é realizado uma cópia do arquivo de porosidade baseline.
# A saída da execução é redirecionada ao final para um arquivo chamado log.log

CONFIG_FILE='area1_config.yaml'
BASELINE_POR_FILE='/home/daniel/Documentos/Git/petroiageo/data/ANP/processed/area1/porosity_data_baseline.h5'
#This should be the same as the starting_porosity_cube_path param in the CONFIG_FILE
TARGET_POR_FILE='/home/daniel/Documentos/Git/petroiageo/data/ANP/processed/area1/porosity_data.h5'

#Create a copy of the baseline
cp $BASELINE_POR_FILE $TARGET_POR_FILE

IT_START=1
IT_END=3

LOG_FILE_NAME=log.log
rm -f $LOG_FILE_NAME
for it in $(seq $IT_START $IT_END)
do
    python3 -u main.py --it $it --config  $CONFIG_FILE --nf 1 --nsf 2 --ntf 3 >> $LOG_FILE_NAME
done