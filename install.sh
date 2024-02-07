#!/usr/bin/bash

function exit_if_failure () {
    if [ $1 -ne 0 ]; then
        echo "[LOG] $2 ABORTED BECAUSE ERROR CODE $?. See above errors for more"
        exit 1
    fi
}

# For using the $(!!) at exit_if_failure calls
set -o history -o histexpand

#As we are at the project's dir, go back one time
PROJECT_DIR=$(pwd)
cd ..

#HDF5 STUFF

CURR_STAGE="HDF5"
#If the hdf5 dir exists, we supose that we 
#can skip the hdf5 clone and install
if [ -e "hdf5" ]; then
    echo "[LOG] FOUND hdf5 DIR AT $(pwd)"' ! Skipping hdf5 clone!'
else
    echo "[LOG] COULD NOT FIND hdf5 DIR at $(pwd)"' ! CLONING...'
    git clone https://github.com/HDFGroup/hdf5.git
    echo '[LOG] CLONING COMPLETE!'
fi

cd "hdf5"
HDF5_PATH=$(pwd)
echo "[LOG] HDF5_PATH: $HDF5_PATH"

echo "[LOG] INSTALLING $CURR_STAGE..."

git checkout hdf5-1_12_2-3-rc1
autoconf
exit_if_failure $? "$(!!)"
./configure --enable-parallel --enable-shared --prefix=$HDF5_PATH/build
exit_if_failure $? "$(!!)"
make -j8
exit_if_failure $? "$(!!)"
make install
exit_if_failure $? "$(!!)"

echo "[LOG] $CURR_STAGE INSTALL COMPLETE"

# Go back again
cd ..

#H5PY STUFF

CURR_STAGE="H5PY"
echo "[LOG] INSTALLING PYTHON DEPENDENCIES FOR $CURR_STAGE"
python3 -m pip uninstall mpi4py
python3 -m pip install mpi4py==3.1.3
python3 -m pip install wheel Cython==3.0.0a11 numpy
echo "[LOG] PYTHON DEPENDENCIES FOR $CURR_STAGE INSTALL COMPLETE"

# If the h5py exists, we supose its already installed
if [ -e "h5py" ]; then
    echo "[LOG] FOUND h5py DIR AT $(pwd) "'! Skipping h5py clone and install!'
else
    echo "[LOG] COULD NOT FIND h5py DIR at $(pwd)"' ! CLONING...'
    git clone https://github.com/h5py/h5py.git
    echo '[LOG] CLONING COMPLETE!'
fi

cd "h5py"

echo "[LOG] INSTALLING $CURR_STAGE...."

git checkout 3.7.0
export CC=mpicc; export HDF5_MPI="ON"; export HDF5_DIR="$HDF5_PATH/build"; python3 -m pip install --no-build-isolation .
exit_if_failure $? "$(!!)"

echo "[LOG] $CURR_STAGE INSTALL COMPLETE"'!'

#PYTHON REQUIREMENTS

CURR_STAGE="PROJECT PYTHON REQUIREMENTS"
echo "[LOG] INSTALLING $CURR_STAGE"
cd $PROJECT_DIR

python3 -m pip install -r requirements

exit_if_failure $? "$(!!)"

echo "[LOG] $CURR_STAGE INSTALL COMPLETE"'!'

echo '[LOG] FULL INSTALL COMPLETE!'
