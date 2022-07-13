import numpy as np
import sys
from io import StringIO
import time
import pandas as pd
import warnings
import common

MAX_DEPTH = 251  # Starts from 1

def gen_expanded_points(main_df, canal_df, real_wells, it):
    # Set distance ring to be generated
    ring = it + 1
    ring_circunf = ring * 2 + 1  # single width
    ring_circunf = ring_circunf**2 - (ring_circunf - 2)**2

    # Allocate ndarray for new points
    # Number of cols = 6 : (x,y,z,well,real,phi)
    expanded_points_np = np.empty((ring_circunf * MAX_DEPTH, 5),
                                  dtype=np.int32)
    expanded_points_phi_np = np.empty((ring_circunf * MAX_DEPTH, ),
                                      dtype=np.float64)

    # Expand around each original well
    well_id = 0
    t1 = time.time()
    for well in real_wells:
        expanded_points_i = 0

        print(f'==== Expanding well {well}')
        for z in range(MAX_DEPTH):
            # Only expand from real points (present at a certain depth)
            well_coord = (well[0], well[1], z)
            is_real_point = True
            if not main_df.index.isin([well_coord]).any():
                is_real_point = False

            for i in range(-ring, ring + 1):
                for j in range(-ring, ring + 1):
                    # Only add ring frontier points
                    if abs(i) == ring or abs(j) == ring:
                        x = well[0] + i
                        y = well[1] + j

                        # Create new expanded point (real=2) with
                        # phi val from canal point
                        canal_phi = canal_df.loc[(x, y, z)]['phi']

                        if canal_phi == 0 or not is_real_point:
                            expanded_points_np[expanded_points_i] = (x, y, z,
                                                                 well_id, 3)
                            expanded_points_phi_np[expanded_points_i] = 0
                        else:
                            expanded_points_np[expanded_points_i] = (x, y, z,
                                                                     well_id, 2)
                            expanded_points_phi_np[expanded_points_i] = canal_phi
                        
                        expanded_points_i = expanded_points_i + 1

        # # Filter the zero values [0.0, 0.0, ... 0.0]
        # # They come from non-expanded points which are not present on the canal
        # filt_expanded_points_np = expanded_points_np[:expanded_points_i]
        # filt_expanded_points_phi_np = expanded_points_phi_np[:
        #                                                      expanded_points_i]

        # Add new points from current well to the main DataFrame
        expanded_points_df = pd.DataFrame(
            # filt_expanded_points_np,
            expanded_points_np,
            columns=['x', 'y', 'z', 'well', 'real'],
            dtype=np.int32)
        expanded_points_df['phi'] = expanded_points_phi_np
        index = pd.MultiIndex.from_arrays([
                                            expanded_points_df['x'], expanded_points_df['y'],
                                            expanded_points_df['z']
                                            ],
                                            names=common.MAIN_DF_INDEX_NAMES
                                        )
        expanded_points_df.set_index(index, inplace=True)
        main_df = pd.concat([main_df, expanded_points_df])

        well_id = well_id + 1

    t2 = time.time()

    return main_df


if __name__ == '__main__':
    data_aug(sys.argv[1], 3)