#!/bin/bash
#SBATCH --nodes=1   	    #Numero de Nós
#SBATCH -p ict_cpu		    #Fila (partition) a ser utilizada
#SBATCH -J feats_calc	    #Nome job
#SBATCH --time=4:00:00
#SBATCH --account=petrobrasiageo
#SBATCH --exclusive		    #Utilização exclusiva dos nós
#SBATCH --no-requeue	    # Não ressubmete um job se ele der erro

module load python/3.9.6

# When using a virtual environment, which is highly recommended
VENV_PATH=../../../../venv
source $VENV_PATH/bin/activate
python3 -m pip list -v

OUTPUT=../../../../data/ANP/v2/area1/npy
FILENAME=../../../../../dados/ANP/pre_stack/area_1_NEAR.npy
WINDOW_1D=9
ALGS=all
N_CPUS=-1

make_features.py -o $OUTPUT -w1 $WINDOW_1D -n_cpu $N_CPUS $FILENAME $ALGS