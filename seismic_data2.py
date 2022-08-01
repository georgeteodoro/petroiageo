import numpy as np
# import pandas as pd
import dask.dataframe as dd
# from numba import jit

import sys

# # Prod of tuple (numba does not allows math.prod)
# @jit(nopython=True)
# def prod(shape):
#     p = 1
#     for s in shape:
#         p = p * s
#     return p

# @jit(nopython=True)
# def get_multiindex_np(shape):
#     # Output arrays initialization
#     # Fields are [x, y, z] and [seismic_data...]
#     num_lines = prod(shape)
#     coords_output = np.empty((num_lines, 3), dtype=np.int32)
#     for x in range(shape[0]):
#         for y in range(shape[1]):
#             for z in range(shape[2]):
#                 # Assign value and increment index
#                 i = x * shape[1] * shape[2] + y * shape[2] + z
#                 coords_output[i] = [int(x), int(y), int(z)]

#     return coords_output


# Get all seismic data on the order the input filenames are passed
def get_all_seismic_data(seismic_columns):

    # Generate filenames
    filenames = [f'./dados/{col}.npy' for col in seismic_columns]

    # Get the shape/dimensions of the 3D hypercube
    hypercube_shape = np.load(filenames[0]).shape

    # Prepare the DataFrame of features
    seismic_nps = np.array([np.load(f).flatten()
                            for f in filenames]).transpose()
    features_ddf = dd.from_array(seismic_nps, columns=seismic_columns)

    # Setup index
    index = np.array(list(range(seismic_nps.shape[0])))
    features_ddf['index'] = dd.from_array(index)
    features_ddf = features_ddf.set_index('index')

    # Returns the persisted operations on memory (i.e., load from file only once)
    return features_ddf.persist(), hypercube_shape


if __name__ == '__main__':
    get_all_seismic_data(['NEAR', 'MID', 'FAR', 'UFAR', 'GERSZ', 'GST'])
