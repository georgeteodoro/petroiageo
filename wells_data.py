import numpy as np
import pandas as pd
from numba import jit


# Converter to ease usage of txt files with numpy efficiency
# This only needs to be ran once
def get_np_wells_data(filename):
    f = open(filename, 'r')
    num_lines = sum(1 for _ in f)
    f.seek(0)  # Reset file pointer

    # Shapes are [x, y, z, well, real] and [phi]
    # Types are [int, int, int, int, int] and [float]
    int_values_np = np.empty(shape=(num_lines, 5), dtype=np.int32)
    phi_np = np.empty(shape=(num_lines, 1), dtype=np.float64)
    i = 0
    real_id = 0  # TODO: ID should start at 0 or 1?
    prev_xy = (-1, -1)

    # Convert wells data to np.array format
    for line in f.readlines():
        fields = [s.replace('\n', '') for s in line.split(' ')]
        if prev_xy != (int(fields[0]), int(fields[1])):
            real_id = real_id + 1
        prev_xy = (int(fields[0]), int(fields[1]))

        # Each point is set as a real point with ID i
        int_values_np[i] = [
            int(fields[0]),
            int(fields[1]),
            int(fields[2]),
            int(real_id),
            int(0)
        ]
        phi_np[i] = float(fields[3])
        i = i + 1

    f.close()

    return int_values_np, phi_np


def merge_wells_data(seismic_df, filename):
    # Open porosity file
    int_values_np, phi_np = get_np_wells_data(filename)

    # Convert wells data to pandas.DataFrame format
    int_values_df = pd.DataFrame(int_values_np,
                                 columns=['x', 'y', 'z', 'well', 'real'])
    phi_df = pd.DataFrame(phi_np, columns=['phi'])
    wells_df = pd.concat([int_values_df, phi_df], axis=1)

    # Creates a join on left (seismic_df)
    result = pd.merge(seismic_df, wells_df, on=['x', 'y', 'z'], how='left')
    result.fillna({'well': -1, 'real': 2}, inplace=True)
    result = result.astype({'well': int, 'real': int})
    # result.set_index('real', inplace=True)

    # Ok to sort since we access points directly, without adding more points
    # However, should we index this value?
    result.sort_values(by='real', ascending=True, inplace=True)

    return result
