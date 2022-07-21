import numpy as np
import pandas as pd
import dask.dataframe as dd
from numba import jit

import sys


# Prod of tuple (numba does not allows math.prod)
@jit(nopython=True)
def prod(shape):
    p = 1
    for s in shape:
        p = p * s
    return p


@jit(nopython=True)
def get_multiindex_np(shape):
    # Output arrays initialization
    # Fields are [x, y, z] and [seismic_data...]
num_lines = prod(shape)
coords_output = np.empty((num_lines, 3), dtype=np.int32)
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            # Assign value and increment index
            i = x * shape[1] * shape[2] + y * shape[2] + z
            coords_output[i] = [int(x), int(y), int(z)]

    return coords_output


# Get all seismic data on the order the input filenames are passed
def get_all_seismic_data(seismic_columns):

    # Generate filenames
    filenames = [f'./dados/{col}.npy' for col in seismic_columns]

    # Load each numpy array
    arr = np.load(filenames[0])
    arr_shape = arr.shape

    # Prepare index array
    coords_np = get_multiindex_np(arr_shape)

    # Prepare the DataFrame
    seismic_np = np.array([np.load(f) for f in filenames]).transpose()
    fetures_df = dd.from_array(seismic_np, columns=seismic_columns)

    # Set coordinates as the index
    index = pd.MultiIndex.from_arrays(coords_np)
    fetures_df.set_index(index, inplace=True)
    fetures_df.sort_index(inplace=True)

    return fetures_df


if __name__ == '__main__':
    get_all_seismic_data(['NEAR', 'MID', 'FAR', 'UFAR', 'GERSZ', 'GST'])
