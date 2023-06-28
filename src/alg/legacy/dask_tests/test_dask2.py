import numpy as np
import dask.dataframe as dd
from timeit import timeit
from math import prod, ceil, floor
from time import time
from numba import jit, prange
from dask.distributed import Client, LocalCluster, wait, progress
from dask.diagnostics import ProgressBar
import dask.config
import logging
import gc

from concurrent.futures import ProcessPoolExecutor

import os, psutil, cProfile

import util
import update_funcs

# 2 workers, 10 threads and 18 GB
# time to beat: 272.35518050193787
# new time:     131.65763115882874

# parallel numba 1 worker, 10 threads and 28 GB
# time: 130.832

# parallel numba 4 workers, 2 threads and 8 GB
# time: 119.66355466842651

# parallel numba 4 workers, 1 threads and 8 GB
# time: 84.23432183265686

# parallel numba, 8 worker, 1 threads and 5 GB
# time: 54.53365516662598


def main():
    # client = update_funcs.initialize_dask(w=1, t=2, mem=16)
    update_funcs.initialize_dask(w=1, t=2, mem=1, disk=40)

    # ddf1, features_ddf, shape, chunksize = update_funcs.load_ddfs('small')
    ddf1, features_ddf, shape, chunksize = update_funcs.load_ddfs("100M")

    t0 = time()
    base_feature_name = "feature1"
    feature_name1 = "feature10"
    feature1 = (-1, 0, 1)
    feature_name2 = "feature20"
    feature2 = (2, 3, -2)

    ddf1[feature_name1] = ddf1.map_partitions(
        update_funcs.update_part,
        update_funcs.Wrapper(features_ddf),
        shape,
        feature1,
        base_feature_name,
        chunksize,
        align_dataframes=False,
        meta=(None, int),
    )

    # ddf1.visualize(filename='outside_part_graph.svg')
    with ProgressBar(), dask.config.set(num_workers=16, scheduler="processes"):
        ddf1 = ddf1.persist()
    t2 = time()
    print(f"=============prepared partitions update {t2-t0}")

    # progress(ddf1)
    print("")
    # ddf1.to_parquet('out_ddf.parquet')

    t3 = time()
    print(f"=============done {t3-t2}")
    # print(ddf1.compute())

    # ddf1[feature_name2] = ddf1.map_partitions(update_funcs.update_part,
    #                                           update_funcs.Wrapper(features_ddf),
    #                                           shape,
    #                                           feature2,
    #                                           base_feature_name,
    #                                           chunksize,
    #                                           align_dataframes=False,
    #                                           meta=(None, int))

    features_min = features_ddf.index.map_partitions(min).compute().to_numpy()
    features_max = features_ddf.index.map_partitions(max).compute().to_numpy()

    ddf1[feature_name2] = ddf1.map_partitions(
        update_funcs.update_part_no_minmax,
        update_funcs.Wrapper(features_ddf),
        features_min,
        features_max,
        shape,
        feature2,
        base_feature_name,
        chunksize,
        align_dataframes=False,
        meta=(None, int),
    )

    with ProgressBar():
        ddf1 = ddf1.persist()
    # progress(ddf1)

    print(ddf1.compute())


if __name__ == "__main__":
    main()
