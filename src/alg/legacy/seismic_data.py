import numpy as np
import pandas as pd
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
def get_single(arr, shape, with_coords=False):
    # Output arrays initialization
    # Fields are [x, y, z] and [seismic_data...]
    num_lines = prod(shape)
    if with_coords:
        coords_output = np.empty((num_lines, 3), dtype=np.int32)
    else:
        coords_output = None
    seismic_output = np.empty((num_lines,), dtype=np.float64)
    i = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Assign value and increment index
                if with_coords:
                    coords_output[i] = [int(x), int(y), int(z)]
                seismic_output[i] = arr[x, y, z]
                i = i + 1

    return coords_output, seismic_output


# Merge all seismic data
# Outputs are generated separately to enforce types (avoid x,y,z with float)
# while enabling sparse allocation of numpy arrays. Using structured arrays
# results in out-of-memory errors.
@jit(nopython=True)
def get_all(arrs, shape):
    # Output arrays initialization
    # Fields are [x, y, z] and [seismic_data...]
    num_lines = prod(shape)
    coords_output = np.empty((num_lines, 3), dtype=np.int32)
    seismic_output = np.empty((num_lines, len(arrs)), dtype=np.float64)
    i = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Create a line with [seismic_data...]
                line_vals = np.empty(len(arrs))
                for j in range(len(arrs)):
                    line_vals[j] = arrs[j][x, y, z]

                # Assign line and increment index
                coords_output[i] = [int(x), int(y), int(z)]
                seismic_output[i] = line_vals
                i = i + 1

    return coords_output, seismic_output


# Get all seismic data on the order the input filenames are passed
def get_all_seismic_data(seismic_columns):
    # Generate filenames
    filenames = [f"./dados/{col}.npy" for col in seismic_columns]

    # Load each numpy array
    arr = np.load(filenames[0])
    arr_shape = arr.shape

    # Get first file and prepare DataFrame
    coords_np, seismic_np = get_single(arr, arr_shape, True)
    coords_df = pd.DataFrame(coords_np, columns=["x", "y", "z"])
    seismic_df = pd.DataFrame(seismic_np, columns=[seismic_columns[0]])
    fetures_df = pd.concat([coords_df, seismic_df], axis=1)

    del coords_np
    del seismic_np
    del seismic_df
    del coords_df

    i = 1  # Column ID for getting the right col name
    for f in filenames[1:]:
        seismic_np = []
        arr = np.load(f)
        # Load single file
        _, seismic_np = get_single(arr, arr_shape)
        del arr
        del _

        seismic_df = pd.DataFrame(seismic_np, columns=[seismic_columns[i]])
        del seismic_np
        i = i + 1

        # add results to main DataFrame
        fetures_df = pd.concat([fetures_df, seismic_df], axis=1)
        del seismic_df

    # Set coordinates as the index
    index = pd.MultiIndex.from_arrays(
        [fetures_df["x"], fetures_df["y"], fetures_df["z"]]
    )
    fetures_df.set_index(index, inplace=True)
    del index
    fetures_df.sort_index(inplace=True)

    return fetures_df


if __name__ == "__main__":
    get_all_seismic_data(["NEAR", "MID", "FAR", "UFAR", "GERSZ", "GST"])
