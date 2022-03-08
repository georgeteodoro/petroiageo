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
    ring_circunf = ring_circunf ** 2 - (ring_circunf - 2) ** 2

    # Allocate ndarray for new points
    # Number of cols = 6 : (x,y,z,well,real,phi)
    expanded_points_np = np.empty((ring_circunf * MAX_DEPTH, 6),
                                  dtype=np.int32)

    # Get all phi values for direct access
    phi_vals = main_df['phi']

    # Expand around each original well
    well_id = 0
    t1 = time.time()
    for well in real_wells:
        expanded_points_i = 0

        for z in range(MAX_DEPTH):
            # Only expand points which have at least 0.05 porosity
            well_coord = (well[0], well[1], z)
            if not main_df.index.isin([well_coord]).any():
                continue
            # Obs: Line bellow will brake the application once there are
            # duplicated points on main_df. This will happen on later
            # expansions for high iteration values of main.py
            if phi_vals.loc[well_coord] <= 0.05:
                continue

            for i in range(-ring, ring + 1):
                for j in range(-ring, ring + 1):
                    # Only add ring frontier points
                    if abs(i) == ring or abs(j) == ring:
                        x = well[0] + i
                        y = well[1] + j

                        # Accumulate phi values of all surrounding points
                        # of (x,y,z) on the window [-avg_window,avg_window]
                        avg_window = 1
                        neighb_phi = 0
                        count = 0
                        for ii in range(-avg_window, avg_window + 1):
                            for jj in range(-avg_window, avg_window + 1):
                                # Don't access points without phi values
                                if i + ii < -ring + 1 or i + ii > ring - 1:
                                    continue
                                if j + jj < -ring + 1 or j + jj > ring - 1:
                                    continue
                                neighb_phi = neighb_phi + phi_vals.loc[x + ii,
                                                                       y + jj,
                                                                       z]
                                count = count + 1

                        # Create new expanded point (real=2) with
                        # phi val from adjacent coordinate
                        expanded_points_np[expanded_points_i] = (x, y, z,
                                                                 well_id, 2,
                                                                 neighb_phi /
                                                                 count)
                        expanded_points_i = expanded_points_i + 1

        # Filter the zero values [0.0, 0.0, ... 0.0]
        # They come from non-expanded points due to low porosity value
        filt_expanded_points_np = expanded_points_np[:expanded_points_i]

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