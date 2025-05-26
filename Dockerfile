FROM python:3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/openmpi-src \
    && cd /tmp/openmpi-src \
    && wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz \
    && tar -xzf openmpi-4.1.6.tar.gz \
    && cd openmpi-4.1.6 \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) all \
    && make install \
    && ldconfig \
    && cd / \
    && rm -rf /tmp/openmpi-src

RUN python -m pip install --upgrade pip setuptools wheel numpy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3-dev \           
    pkg-config \             
    && rm -rf /var/lib/apt/lists/*

ENV CC=mpicc \
    HDF5_MPI=ON \
    HDF5_DIR=/hdf5/build

RUN git clone https://github.com/HDFGroup/hdf5.git hdf5 \
    && cd hdf5 \
    && git checkout hdf5-1_12_2-3-rc1 \
    && autoconf \
    && ./configure --enable-parallel --enable-shared --prefix=/hdf5/build \
    && make -j8 \
    && make install

RUN pip3 uninstall -y mpi4py \
    && pip3 install mpi4py==4.0.1 \
    && git clone https://github.com/h5py/h5py.git h5py \
    && cd h5py \
    && git checkout 3.11.0 \
    && pip3 install wheel Cython==3.1.0a1 \
    && export CC=mpicc; export HDF5_MPI="ON"; export HDF5_DIR="/hdf5/build"; pip3 install --no-build-isolation .
