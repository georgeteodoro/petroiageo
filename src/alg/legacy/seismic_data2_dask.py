import numpy as np
import dask.dataframe as dd
from dask.distributed import progress
from math import prod
import os, psutil

import dask_utils


# Get all seismic data on the order the input filenames are passed
def get_all_seismic_data(seismic_columns):
    # process = psutil.Process(os.getpid())
    # print(f'init mem: {dask_utils.sizeof_fmt(process.memory_info().rss)}')

    # Generate filenames
    filenames = [f"./dados/{col}.npy" for col in seismic_columns]

    # Get the shape/dimensions of the 3D hypercube
    hypercube_shape = np.load(filenames[0]).shape

    # Prepare the DataFrame of features
    print("Loading npy")
    seismic_nps = np.array(
        [np.load(f).flatten() for f in filenames]
    ).transpose()

    print("Setting dask dataframe")
    print(f"seismic_nps shape: {seismic_nps.shape}")
    chunksize = 10_000_000
    features_ddf = dd.from_array(
        seismic_nps, columns=seismic_columns, chunksize=chunksize
    )
    print(features_ddf)
    features_ddf = features_ddf.persist()
    print("")
    rows = seismic_nps.shape[0]
    del seismic_nps

    # Create coordinates values
    print("Creating coordinates")
    xs_np = np.zeros(prod(hypercube_shape), dtype=int)
    ys_np = np.zeros(prod(hypercube_shape), dtype=int)
    zs_np = np.zeros(prod(hypercube_shape), dtype=int)
    ii = 0
    for i in range(hypercube_shape[0]):
        for j in range(hypercube_shape[1]):
            for k in range(hypercube_shape[2]):
                xs_np[ii] = int(i)
                ys_np[ii] = int(j)
                zs_np[ii] = int(k)
                ii = ii + 1

    print("Setting coordinates")
    features_ddf["x"] = dd.from_array(xs_np, chunksize=chunksize)
    features_ddf["y"] = dd.from_array(ys_np, chunksize=chunksize)
    features_ddf["z"] = dd.from_array(zs_np, chunksize=chunksize)

    features_ddf = features_ddf.persist()
    print("")
    del xs_np
    del ys_np
    del zs_np

    # Setup index
    print("Setting index")
    index = np.array(list(range(rows)), dtype=int)
    features_ddf["index"] = dd.from_array(index, chunksize=chunksize)
    features_ddf = features_ddf.set_index("index")

    features_ddf = features_ddf.persist()
    print("")
    print(features_ddf)

    # Returns the persisted operations on memory (i.e., load from file only once)
    return features_ddf, hypercube_shape


if __name__ == "__main__":
    features, hypercube_shape = get_all_seismic_data(
        ["NEAR", "MID", "FAR", "UFAR", "GERSZ", "GST"]
    )

    features.to_parquet("./dados/seismic_features.parquet")
    # print(features.compute())
