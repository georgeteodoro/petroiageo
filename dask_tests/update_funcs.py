import numpy as np
import dask.dataframe as dd
from timeit import timeit
from math import prod, ceil, floor
from time import time
from numba import jit, prange, config
import numba as nb
from dask.distributed import Client, LocalCluster, wait, progress
import dask.config
import logging
import gc

import util

base_feature_name = 'feature1'
feature_name1 = 'feature10'
feature1 = (-1, 0, 1)
feature_name2 = 'feature20'
feature2 = (2, 3, -2)

# (sufix: (shape, partitions))
ddfs_desc = {
    'small': ((2, 1, 5), 3),
    '100M': ((200, 1000, 500), 1000),
    '500M': ((500, 2000, 500), 1000)
}

# set_num_threads(4)
# config.THREADING_LAYER = 'omp'

import ctypes


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def initialize_dask(w, t, mem, disk):
    virtual_mem = (mem + disk) / w
    spill = (mem / w) / virtual_mem

    dask.config.set({
        'distributed.worker.memory.target': None,
        'distributed.worker.memory.spill': spill,
        'distributed.worker.memory.pause': spill + 0.1,
        # 'distributed.worker.memory.terminate': False
    })
    # cluster = LocalCluster(
    #     n_workers=w,
    #     threads_per_worker=t,
    #     memory_limit=f'{virtual_mem}GB',
    #     processes=True,
    #     silence_logs=logging.ERROR,
    # )
    # client = Client(cluster)

    # return client


def load_ddfs(name):
    shape = ddfs_desc[name][0]
    npartitions = ddfs_desc[name][1]

    size = int(prod(shape))
    chunksize = int(size / npartitions)

    ddf = dd.read_parquet('data/ddf1-' + name + '.parquet')
    features_ddf = dd.read_parquet('data/features_ddf-' + name + '.parquet')

    return ddf, features_ddf, shape, chunksize


def repartition(ddf1, features_ddf, npartitions):
    ddf1 = ddf1.repartition(npartitions=npartitions)
    ddf1 = ddf1.persist()

    features_ddf = features_ddf.repartition(npartitions=npartitions)
    features_ddf = features_ddf.persist()

    wait(features_ddf)

    return ddf1, features_ddf, int(len(ddf1) / npartitions)


###################################################################


# Return locations of points to be read
@jit(nopython=True, parallel=True)
def get_loc2(part_np, feature, shape, chunksize):
    loc_np = np.empty((len(part_np)), dtype=np.int64)
    part_loc_np = np.empty((len(part_np)), dtype=np.int64)
    for i in prange(len(part_np)):
        # print(type(part_np[i]))
        # print(part_np[i])
        newx = min(max(part_np[i, 1] + feature[0], 0), shape[0] - 1)
        newy = min(max(part_np[i, 2] + feature[1], 0), shape[1] - 1)
        newz = min(max(part_np[i, 3] + feature[2], 0), shape[2] - 1)
        loc_id = newx * shape[2] * shape[1] + newy * shape[2] + newz

        loc_np[i] = loc_id
        part_loc_np[i] = floor(loc_id / chunksize)

    return loc_np, part_loc_np


@jit(nopython=True, parallel=True)
def update_given_part_np(part_id_np, part_valid_f_np, feature_part_np,
                         min_f_id, max_f_id, chunksize):
    out_np = np.empty((len(part_id_np)), dtype=np.int64)

    for i in prange(len(part_id_np)):
        if (part_id_np[i] >= min_f_id) & (part_id_np[i] <= max_f_id):
            out_np[i] = feature_part_np[part_id_np[i] % chunksize]
        else:
            out_np[i] = part_valid_f_np[i]

    return out_np


###################################################################


# This should work better when using ghost-zones
def update_part(part, features_w, shape, feature, feature_name, chunksize):
    trim_memory()

    loc_np, part_loc_np = get_loc2(part.to_numpy(), feature, shape, chunksize)

    # Get list of partitions required to lookup data
    part_to_fill = set(part_loc_np)
    del part_loc_np
    gc.collect()
    trim_memory()

    out_np = np.empty((len(part)), dtype=np.int64)

    for p in part_to_fill:
        feature_part = features_w.ddf.get_partition(p)
        feature_part_np = feature_part[feature_name].compute().to_numpy()
        out_np = update_given_part_np(loc_np, out_np, feature_part_np,
                                      feature_part.index.min().compute(),
                                      feature_part.index.max().compute(),
                                      chunksize)

    return out_np


###################################################################


# Improved performance on larger pipelines (more tasks)
# Improvement felt from 1000 partitions onward
def update_part_no_minmax(part, features_w, features_min, features_max, shape,
                          feature, feature_name, chunksize):
    trim_memory()

    loc_np, part_loc_np = get_loc2(part.to_numpy(), feature, shape, chunksize)

    # Get list of partitions required to lookup data
    part_to_fill = set(part_loc_np)
    del part_loc_np
    gc.collect()
    trim_memory()

    out_np = np.empty((len(part)), dtype=np.int64)

    for p in part_to_fill:
        feature_part = features_w.ddf.get_partition(p)[feature_name]
        feature_part_np = feature_part.compute().to_numpy()
        out_np = update_given_part_np(loc_np, out_np, feature_part_np,
                                      features_min[int(p)],
                                      features_max[int(p)], chunksize)

    return out_np


# Wrapper of a dask df to avoid it being transformed into a pandas df
# This allows to retain partition information
class Wrapper(object):

    def __init__(self, ddf):
        self.ddf = ddf