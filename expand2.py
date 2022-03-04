import numpy as np
import sys
from io import StringIO
import time
import pandas as pd
import warnings

MAX_DEPTH = 251  # Starts from 1


def gen_expanded_points(main_df, real_wells, it):
    # Set distance ring to be generated
    ring = it + 1
    ring_circunf = ring * 2 + 1  # single width
    ring_circunf = 2 * ring_circunf + 2 * (ring_circunf - 2)

    # Allocate ndarray for new points
    # Number of cols = 6 : (x,y,z,well,real,phi)
    expanded_points_np = np.empty((ring_circunf * MAX_DEPTH, 6),
                                  dtype=np.int32)

    # Expand around each original well
    well_id = 0
    t1 = time.time()
    for well in real_wells:
        ii = 0

        for z in range(MAX_DEPTH):
            # Only expand points which have at least 0.05 porosity
            well_coord = (well[0], well[1], z)
            if not main_df.index.isin([well_coord]).any():
                continue
            cur_phi = main_df.loc[well_coord, 'phi']
            if cur_phi <= 0.05:
                continue

            for i in range(-ring, ring + 1):
                for j in range(-ring, ring + 1):
                    # Only add ring frontier points
                    if abs(i) == ring or abs(j) == ring:
                        x = well[0] + i
                        y = well[1] + j

                        # Create new expanded point (real=2) with empty phi val
                        expanded_points_np[ii] = (x, y, z, well_id, 2, 0)
                        ii = ii + 1

        # Filter the zero values [0.0, 0.0, ... 0.0]
        # They come from non-expanded points due to low porosity value
        filt_expanded_points_np = expanded_points_np[:ii]

        # Add new points to DataFrame
        expanded_points_df = pd.DataFrame(
            filt_expanded_points_np,
            columns=['x', 'y', 'z', 'well', 'real', 'phi'],
            dtype=np.int32)
        index = pd.MultiIndex.from_arrays([
            expanded_points_df['x'], expanded_points_df['y'],
            expanded_points_df['z']
        ])
        expanded_points_df.set_index(index, inplace=True)
        main_df = pd.concat([main_df, expanded_points_df])

        # print(f'[expand] done with point {well}')
        # print(main_df)

        well_id = well_id + 1

    t2 = time.time()
    # print(f'[expand] total time: {t2-t1}')

    return main_df


if __name__ == '__main__':
    data_aug(sys.argv[1], 3)