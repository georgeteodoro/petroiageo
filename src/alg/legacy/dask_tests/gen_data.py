import numpy as np
import dask.dataframe as dd
from math import prod, ceil
from numba import jit, prange
from dask.distributed import Client, LocalCluster
import logging

MAX_DASK_PARTITION_SIZE = 50000


# Dask can only initialize 50000 rows per dd.from_array
# It drops rows beyond this
def append(np_array, col_name):
    max_i = ceil(len(np_array) / MAX_DASK_PARTITION_SIZE)
    lists = []
    for i in range(max_i):
        lists = lists + [
            dd.from_array(np_array[MAX_DASK_PARTITION_SIZE *
                                   i:MAX_DASK_PARTITION_SIZE * (i + 1)],
                          columns=[col_name],
                          chunksize=MAX_DASK_PARTITION_SIZE)
        ]
    return dd.concat(lists)


def main():
    cluster = LocalCluster(n_workers=1,
                       threads_per_worker=8,
                       memory_limit='28GB',
                       processes=True,
                       silence_logs=logging.ERROR)
    client = Client(cluster)

    # sufix = 'small'
    # shape = (2, 1, 5)
    # npartitions = 3
    
    # sufix = '100M'
    # shape = (200, 1000, 500)
    # npartitions = 1000

    sufix = '500M'
    shape = (5000, 2000, 500)
    npartitions = 1000
    
    feature_name = 'feature1'

    size = int(prod(shape))
    chunksize = int(size / npartitions)

    # xs_np = np.zeros(size, dtype=int)
    # ys_np = np.zeros(size, dtype=int)
    # zs_np = np.zeros(size, dtype=int)
    # ii = 0
    # for i in range(shape[0]):
    #     for j in range(shape[1]):
    #         for k in range(shape[2]):
    #             xs_np[ii] = int(i)
    #             ys_np[ii] = int(j)
    #             zs_np[ii] = int(k)
    #             ii = ii + 1

    # @jit(nopython=True, parallel=True)
    @jit(nopython=True)
    def update_vals(np_array, shape, coord):
        ii = 0
        # for i in prange(shape[0]):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if coord == 0:
                        np_array[ii] = np.int64(i)
                    elif coord == 1:
                        np_array[ii] = np.int64(j)
                    elif coord == 2:
                        np_array[ii] = np.int64(k)
                    ii = ii + 1

    index = np.zeros(size, dtype=np.int64)
    update_vals(index, (1,1,size), 2)
    # index = np.array(list(range(size)))
    ddf1 = append(index, 'phi')
    ddf1 = ddf1.persist()
    del index
    print('done phi')

    xs_np = np.zeros(size, dtype=np.int64)
    update_vals(xs_np, shape, np.int64(0))
    ddf1['x'] = append(xs_np, 'x').x
    ddf1 = ddf1.persist()
    del xs_np
    print('done xs_np')

    ys_np = np.zeros(size, dtype=np.int64)
    update_vals(ys_np, shape, np.int64(1))
    ddf1['y'] = append(ys_np, 'y').y
    ddf1 = ddf1.persist()
    del ys_np
    print('done ys_np')

    zs_np = np.zeros(size, dtype=np.int64)
    update_vals(zs_np, shape, np.int64(2))
    ddf1['z'] = append(zs_np, 'z').z
    ddf1 = ddf1.persist()
    del zs_np
    print('done zs_np')

    index = np.zeros(size, dtype=np.int64)
    update_vals(index, (1,1,size), 2)
    ddf1['idx'] = append(index, 'idx').idx
    ddf1 = ddf1.persist()
    del index
    print('done index')

    ddf1 = ddf1.set_index('idx', shuffle='disk', sorted=True, compute=False)
    ddf1 = ddf1.repartition(npartitions=npartitions)
    print(f'size: {size} with {ddf1.npartitions} chunks of size {chunksize}')

    ddf1.to_parquet('ddf1-' + sufix + '.parquet')
    del ddf1
    print('done ddf1')

    index = np.zeros(size, dtype=np.int64)
    update_vals(index, (1,1,size), 2)
    features_ddf = append(index, 'idx')
    features_ddf = features_ddf.persist()
    features_ddf[feature_name] = features_ddf.idx*10
    features_ddf = features_ddf.persist()
    features_ddf = features_ddf.set_index('idx',
                                          shuffle='disk',
                                          sorted=True,
                                          compute=False)
    features_ddf = features_ddf.repartition(npartitions=npartitions)

    features_ddf.to_parquet('features_ddf.parquet')
    features_ddf.to_parquet('features_ddf-' + sufix + '.parquet')
    print('done features_ddf')


if __name__ == '__main__':
    main()