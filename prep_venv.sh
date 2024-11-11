#!/bin/bash

module load anaconda3/2020.02_sequana openmpi/gnu/4.1.6_sequana gcc/13.2_sequana

conda create --prefix=/petrobr/parceirosbr/petrobrasiageo/willian.barreiros/envs/conda-py3.12 python=3.12

# Remove the linker from conda 
mv /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/envs/conda-py3.12/compiler_compat/ld /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/envs/conda-py3.12/compiler_compat/ld_conda

conda activate /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/envs/conda-py3.12

python3 -m venv /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/envs/venv-py3.12/

