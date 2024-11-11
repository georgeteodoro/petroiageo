#!/bin/bash

# Run the venv activate if not running with sudo
# source /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/envs/venv-py3.12/bin/activate

module load openmpi/gnu/4.1.6_sequana

# pip deps
pip3 install -r /petrobr/parceirosbr/petrobrasiageo/willian.barreiros/petroiageo/requirements 
pip3 install mpi4py==4.0.1

# hdf5
git clone https://github.com/HDFGroup/hdf5.git
cd hdf5
git checkout hdf5-1_12_2-3-rc1
autoreconf
./configure --enable-parallel --enable-shared --prefix=<HDF5_PATH>/build
make -j8
make install

# h5py
git clone https://github.com/h5py/h5py.git
cd h5py
git checkout 3.11.0
pip3 install wheel Cython==3.1.0a1
export CC=mpicc; export HDF5_MPI="ON"; export HDF5_DIR="<HDF5_PATH>/build"; pip3 install --no-build-isolation .