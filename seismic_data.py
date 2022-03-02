import numpy as np
import pandas as pd
from numba import jit


# Prod of tuple (numba does not allows math.prod)
@jit(nopython=True)
def prod(shape):
    p = 1
    for s in shape:
        p = p * s
    return p


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
    filenames = [f'./dados/{col}.npy' for col in seismic_columns]

    # Load each numpy array
    first = np.load(filenames[0])
    arrs = np.array([first] + [np.load(f) for f in filenames[1:]])

    # Get separate np arrays and convert them to DataFrames
    coords_np, seismic_np = get_all(arrs, first.shape)
    coords_df = pd.DataFrame(coords_np, columns=['x', 'y', 'z'])
    seismic_df = pd.DataFrame(seismic_np, columns=seismic_columns)

    # Add coordinates to main dataframe
    fetures_df = pd.concat([coords_df, seismic_df], axis=1)

    # Set coordinates as the index
    index = pd.MultiIndex.from_arrays(
        [fetures_df['x'], fetures_df['y'], fetures_df['z']])
    fetures_df.set_index(index, inplace=True)
    fetures_df.sort_index(inplace=True)

    return fetures_df


if __name__ == '__main__':
    get_all_seismic_data([
        './dados/NEAR.npy', './dados/MID.npy', './dados/FAR.npy',
        './dados/UFAR.npy', './dados/GERSZ.npy', './dados/GST.npy'
    ])
