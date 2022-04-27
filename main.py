import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from os import linesep
from mpi4py import MPI

# from memory_profiler import profile

import seismic_data
import wells_data
import expand2
import petro2
import petro_dist
import apply4

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

# Constants
INIT_IT = 2

real_wells = [(134, 227), (146, 500), (167, 186), (174, 365), (200, 102),
              (236, 113), (250, 315), (287, 242), (230, 194), (344, 276)]


def main():

    # Instantiate pandas dataframe for all data
    # Data structure is composed by:
    #   x,y,z(depth),
    #   well => Well ID (-1 if it's not an original real point.)
    #                   (Has the ID from the original real well)
    #                   (from which it was expanded.           )
    #   real => [3=expanded, to be propagated, 2=expanded canal,
    #            1=propagated, 0=real well point]
    #   phi  => Porosity value
    #   rho  => ?
    #   vp   => ?
    #   vs   => ?

    # Read seismic data and add it to a dataframe
    seismic_features_names = ["NEAR", "MID", "FAR", "UFAR", "GERSZ", "GST"]

    # Features which do not need to be expanded on the window
    other_features_names = []

    # # Serialize seismic read for single-node test with mpi
    # features_df = []
    # cur_p = -1
    # while True:
    #     comm.Barrier()
    #     cur_p = cur_p + 1
    #     if cur_p == mpi_size:
    #         break
    #     if rank != cur_p:
    #         continue
    #     print(f'reading rank {rank}')
    print("[main] Loading seismic data")
    features_df = seismic_data.get_all_seismic_data(seismic_features_names +
                                                    other_features_names)

    print("[main] Features DataFrame:")
    print(features_df)

    # Real wells' data into a main dataframe
    print("[main] Loading wells values")
    main_df = wells_data.get_wells_data('./dados/porosity-canal.txt')

    # Separate the main DataFrame into two, one with only canal points
    canal_df = main_df[main_df['real'] == 2]
    main_df = main_df[main_df['real'] != 2]

    # Expand canal_df to have values across the whole hipercube
    # This allows expand to access each point with DataFrame.loc[]
    # instead of using DataFrame.isin() to check whether a canal point
    # is there.
    t1 = time.time()
    full_canal_np = np.zeros(434 * 646 * 251)
    xs_np = np.zeros(434 * 646 * 251)
    ys_np = np.zeros(434 * 646 * 251)
    zs_np = np.zeros(434 * 646 * 251)
    ii = 0
    for i in range(434):
        for j in range(646):
            for k in range(251):
                xs_np[ii] = i
                ys_np[ii] = j
                zs_np[ii] = k
                ii = ii + 1
    for [x, y, z, phi] in canal_df[['x', 'y', 'z', 'phi']].values:
        full_canal_np[int(x) * 646 * 251 + int(y) * 251 + int(z)] = phi
    canal_df = pd.DataFrame(full_canal_np, columns=['phi'])

    # Add index of xyz and sort dataframe for better access times
    print("[main] Indexing all data by (x,y,z)")
    index = pd.MultiIndex.from_arrays(
        [main_df['x'], main_df['y'], main_df['z']])
    main_df.set_index(index, inplace=True)
    main_df.sort_index(inplace=True)

    index = pd.MultiIndex.from_arrays([xs_np, ys_np, zs_np])
    canal_df.set_index(index, inplace=True)
    canal_df.sort_index(inplace=True)
    # print(f'hipercube gen time {t2-t1}')

    # Free indexes np arrays
    xs = None
    ys = None
    zs = None
    full_canal_np = None

    iterations = 10

    # Generate seismic features names
    window = 3
    all_features = other_features_names
    for f in seismic_features_names:
        for i in range(-window, window + 1):
            for j in range(-window, window + 1):
                for k in range(-window, window + 1):
                    all_features.append((f, i, j, k))
    t2 = time.time()

    print(f'[main] Initial data loading time: {t2-t1}')

    print("[main] Main DataFrame [initial]:")
    print(main_df)

    for it in range(iterations):
        t1 = time.time()

        print(f"[main][{it}] Expanding points")
        main_df = expand2.gen_expanded_points(main_df, canal_df, real_wells,
                                              it)
        main_df.to_csv(f'tmp_data/expanded{it}.csv', index=False)
        print(main_df)

        # Info for validating points generation (All is OK!)
        # print(f'[{it}] real: {len(main_df[main_df["real"] == 0])}')
        # print(f'[{it}] propagated: {len(main_df[main_df["real"] == 1])}')
        # print(f'[{it}] expanded-canal: {len(main_df[main_df["real"] == 2])}')
        # print(f'[{it}] expanded-new: {len(main_df[main_df["real"] == 3])}')

        t2 = time.time()

        print(f"[main][{it}] Performing feature selection")
        # Only uses real, previously predicted and expanded canal points
        # for feature selection
        feature_selection_points_df = main_df[main_df['real'] != 3]
        print('[main] Points for feature selection:')
        print(feature_selection_points_df)
        if mpi_size == 1:
            best_features_set, best_error = petro2.get_features_sets(
                feature_selection_points_df, features_df, all_features, 2, 4)
        else:
            best_features_set, best_error = petro_dist.get_features_sets(
                feature_selection_points_df, features_df, all_features, 10, 0)
        # print(features_sets)

        print(f'[main][{it}] Best features set:'\
              f' {best_features_set} with {best_error} error'
        )

        t3 = time.time()

        print(f"[main][{it}] Performing predictions on new expanded points")
        main_df = apply4.perf_predition(best_features_set, main_df,
                                        features_df)
        print(main_df)
        main_df.to_csv(f'tmp_data/predicted{it}.csv', index=False)

        t4 = time.time()
        print(f'[main][times][{it}] total_it_time {t4-t1}')
        print(f'[main][times][{it}] expansion {t2-t1}')
        print(f'[main][times][{it}] feature_selection {t3-t2}')
        print(f'[main][times][{it}] propagation {t4-t3}')


if __name__ == '__main__':
    main()
