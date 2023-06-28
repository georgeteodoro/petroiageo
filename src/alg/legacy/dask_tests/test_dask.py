import numpy as np
import dask.dataframe as dd
from timeit import timeit
from math import prod, ceil, floor
from time import time
from numba import jit, prange
from dask.distributed import Client, LocalCluster
import logging
import gc
import os, psutil

import util


# 500M test results:
# 2 workers with 10 threads and 18 GB
# Dask DataFrame Structure:
#                     phi      x      y      z
# npartitions=1000
#                   int64  int64  int64  int64
#                     ...    ...    ...    ...
# ...                 ...    ...    ...    ...
#                     ...    ...    ...    ...
#                     ...    ...    ...    ...
# Dask Name: read-parquet, 1000 tasks
# Dask DataFrame Structure:
#                  feature1
# npartitions=1000
#                     int64
#                       ...
# ...                   ...
#                       ...
#                       ...
# Dask Name: read-parquet, 1000 tasks
# =============loaded ddfs: 0.03247570991516113

# DDF with feature indices
# =============get features indices: 0.06323099136352539

# Finding the partition on which an outside point is

# DDF with feature values within partition
# =============within partition values: 0.1673901081085205

# DDF with feature values from other partitions
# =============outside partitions update 272.35518050193787


def main():
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=10,
        memory_limit="18GB",
        processes=True,
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    process = psutil.Process(os.getpid())  # to check for mem usage

    t0 = time()
    feature = (-1, 0, 1)

    # sufix = 'small'
    # shape = (2, 1, 5)
    # npartitions = 3

    # sufix = '100M'
    # shape = (200, 1000, 500)
    # npartitions = 1000

    sufix = "500M"
    shape = (500, 2000, 500)
    npartitions = 1000

    feature_name = "feature1"

    size = int(prod(shape))
    chunksize = int(size / npartitions)

    ddf1 = dd.read_parquet("data/ddf1-" + sufix + ".parquet")
    features_ddf = dd.read_parquet("data/features_ddf-" + sufix + ".parquet")

    print(ddf1)
    print(features_ddf)

    t3 = time()
    print(f"=============loaded ddfs: {t3-t0}")

    @jit(nopython=True)
    def get_loc2(part_np, feature, shape):
        out_np = np.empty((len(part_np)), dtype=np.int64)
        i = 0
        for row in part_np:
            newx = min(max(row[1] + feature[0], 0), shape[0] - 1)
            newy = min(max(row[2] + feature[1], 0), shape[1] - 1)
            newz = min(max(row[3] + feature[2], 0), shape[2] - 1)

            out_np[i] = newx * shape[2] * shape[1] + newy * shape[2] + newz
            i = i + 1

        return out_np

    def part_get_loc2(part, feature, shape):
        return get_loc2(part.to_numpy(), feature, shape)

    ddf1["f_idx"] = ddf1.map_partitions(
        part_get_loc2, feature, shape, meta=(None, int)
    )

    print("")
    print("DDF with feature indices")
    ddf1 = ddf1.persist()
    # print(ddf1.compute())

    t5 = time()
    print(f"=============get features indices: {t5-t3}")

    @jit(nopython=True)
    def np_map(part_np, feature_np, min_f_id, max_f_id, chunksize):
        out_np = np.empty((len(part_np)), dtype=np.int64)

        i = 0
        # Values from inside the partition are filled correctly
        # However, values from outside can filled incorrectly with values
        # from the current feature partition. This is solved later on
        # update_given_part, which updates values outside the partition
        # This if makes no measurable time difference vs the version
        # without, which fills the wrong values
        for i in range(len(part_np)):
            id = part_np[i]
            if (id >= min_f_id) & (id <= max_f_id):
                out_np[i] = feature_np[part_np[i] % chunksize]
            else:
                out_np[i] = -2

        return out_np

    # Set rows with indices within the partition
    def update_within_part2(part_df, feature_part, feature_name):
        min_f_id = feature_part.index.min()
        max_f_id = feature_part.index.max()

        return np_map(
            part_df.f_idx.to_numpy(),
            feature_part[feature_name].to_numpy(),
            min_f_id,
            max_f_id,
            chunksize,
        )

    ddf1[feature_name] = -1
    ddf1[feature_name] = ddf1.map_partitions(
        update_within_part2,
        features_ddf,
        feature_name,
        align_dataframes=True,
        meta=(None, int),
    )
    ddf1 = ddf1.persist()

    # print('done persist within')
    # not_comp = len(ddf1[ddf1[feature_name] == -2])
    # print('done len')
    # print(
    #     f'not computed: {not_comp}/{len(ddf1)}:'\
    #     f' {100*not_comp/len(ddf1)}%'
    # )

    print("")
    print("Finding the partition on which an outside point is")

    @jit(nopython=True)
    def update_given_part_np(part_id_np, feature_part_np, chunksize):
        out_np = np.empty((len(part_id_np)), dtype=np.int64)

        i = 0
        for id in part_id_np:
            if feature_part_np[i] == -2:
                out_np[i] = floor(id / chunksize)
            else:
                out_np[i] = -1
            i = i + 1

        return out_np

    # Set rows with indices within the partition
    def get_outside_part(part, feature_name, chunksize):
        return update_given_part_np(
            part.f_idx.to_numpy(), part[feature_name].to_numpy(), chunksize
        )

    ddf1["outside_part"] = -1
    ddf1["outside_part"] = ddf1.map_partitions(
        get_outside_part, feature_name, chunksize, meta=(None, int)
    )
    ddf1 = ddf1.persist()

    # print(ddf1.compute())
    # print('')

    # def get_list_of_outside_part(part):
    #     out_list = []

    #     for p in part:
    #         if p != -1:
    #             out_list = out_list + [p]

    #     return list(set(out_list))

    # @jit(nopython=True)
    # def get_list_of_outside_part2(part):
    #     out_np = np.array((len(part), -1), dtype=np.int64)

    #     i = 0
    #     for p in part:
    #         if p != -1:
    #             out_np[i] = p
    #         i = i + 1

    #     return out_np

    # for p in ddf1.partitions:
    #     # print(get_list_of_outside_part2(p.outside_part.compute().to_numpy()))
    #     print(p.outside_part.compute().to_numpy())
    #     print(list(set(p.outside_part.compute().to_numpy())))

    # return

    print("")
    print("DDF with feature values within partition")
    ddf1 = ddf1.persist()

    t6 = time()
    print(f"=============within partition values: {t6-t5}")

    print("")
    print("DDF with feature values from other partitions")

    @jit(nopython=True)
    def update_given_part_np(
        part_id_np,
        part_valid_f_np,
        feature_part_np,
        min_f_id,
        max_f_id,
        chunksize,
    ):
        out_np = np.empty((len(part_id_np)), dtype=np.int64)

        for i in range(len(part_id_np)):
            if (part_id_np[i] >= min_f_id) & (part_id_np[i] <= max_f_id):
                out_np[i] = feature_part_np[part_id_np[i] % chunksize]
            else:
                out_np[i] = part_valid_f_np[i]

        return out_np

    # # Set rows with indices within the partition
    # def update_given_part2(part, feature_part, feature_name, min_f_id,
    #                        max_f_id, chunksize):
    #     return update_given_part_np(part.f_idx.to_numpy(),
    #                                 part[feature_name].to_numpy(),
    #                                 feature_part[feature_name].to_numpy(),
    #                                 min_f_id, max_f_id, chunksize)

    def update_outside_part(part, features_w, feature_name, chunksize):
        part_to_fill = set(part.outside_part.to_numpy())
        part_to_fill.discard(-1)
        for p in part_to_fill:
            feature_part = features_w.ddf.get_partition(p)
            feature_part_np = feature_part[feature_name].compute().to_numpy()
            part[feature_name] = update_given_part_np(
                part.f_idx.to_numpy(),
                part[feature_name].to_numpy(),
                feature_part_np,
                feature_part.index.min().compute(),
                feature_part.index.max().compute(),
                chunksize,
            )

        return part[feature_name]

    # Wrapper of a dask df to avoid it being transformed into a pandas df
    # This allows to retain partition information
    class Wrapper(object):
        def __init__(self, ddf):
            self.ddf = ddf

    ddf1[feature_name] = ddf1.map_partitions(
        update_outside_part,
        Wrapper(features_ddf),
        feature_name,
        chunksize,
        align_dataframes=False,
        meta=(None, int),
    )
    ddf1 = ddf1.drop(columns=["outside_part"])

    # ddf1.visualize(filename='outside_part_graph.svg')
    ddf1 = ddf1.persist()
    print(ddf1)

    ddf1.to_parquet("out_ddf.parquet")

    t8 = time()
    print(f"=============outside partitions update {t8-t6}")


if __name__ == "__main__":
    main()
