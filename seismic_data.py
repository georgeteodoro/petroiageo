import numpy as np
from numba import jit


# Prod of tuple (numba does not allows math.prod)
@jit(nopython=True)
def prod(shape):
    p = 1
    for s in shape:
        p = p * s
    return p


# Merge all seismic data
@jit(nopython=True)
def get_all(arrs, shape):
    # Output array with its index
    # Fields are [x, y, z, seismic_data...]
    output = np.empty((prod(shape), 3 + len(arrs)))
    i = 0

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                # Create a line with [x, y, z, seismic_data...]
                line_vals = np.empty(len(arrs) + 3)
                line_vals[:3] = [x, y, z]
                for i in range(len(arrs)):
                    line_vals[i + 3] = arrs[i][x, y, z]

                # Assign line and increment index
                output[i] = line_vals
                i = i + 1

    return output


# Get all seismic data on the order the input filenames are passed
def get_all_seismic_data(filenames):

    first = np.load(filenames[0])

    arrs = np.array([first] + [np.load(f) for f in filenames[1:]])
    return get_all(arrs, first.shape)


if __name__ == '__main__':
    get_all_seismic_data([
        "./dados/NEAR.npy", "./dados/MID.npy", "./dados/FAR.npy",
        "./dados/UFAR.npy", "./dados/GERSZ.npy", "./dados/GST.npy"
    ])
