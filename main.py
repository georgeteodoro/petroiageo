import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from os import linesep
from mpi4py import MPI
import sys
import common
import argparse
import concurrent.futures
from enum import Enum, auto

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

real_wells = [(134, 227), (146, 500), (167, 186), (174, 365), (200, 102),
              (236, 113), (250, 315), (287, 242), (230, 194), (344, 276)]


class MPI_TAGS(Enum):
    WORKER_EMPTY_RESULT = auto()  # Signals first ask from worker
    MANAGER_FEATURE_DONE = auto()  # Signals done finding new feature
    MANAGER_FINISH = auto()  # Signals done execution of current iteration


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(main_df,
                      features_df,
                      all_features,
                      parallel_settings,
                      exp_n_features,
                      f_width=0):
    if mpi_size < 2:
        print("[petro-dist] 2 minimum processes required")
        return None

    if rank == manager_rank:
        return petro_dist.manager(all_features, exp_n_features, f_width)
    elif rank != manager_rank:
        return worker(main_df, features_df, parallel_settings)


def worker(main_df, features_df, parallel_settings):
    print(f"[petro-dist][w{rank}]")

    # Create a shallow copy of main_df for adding new columns
    # Data from is main_df is only referenced, not copied
    cur_df = main_df.copy(deep=False)

    cur_f_set = ['x', 'y', 'z']

    # Run jobs until manager finishes
    while True:
        t0 = time.time()

        # Request a job from manager
        comm.send(parallel_settings['n_cpus'],
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)

        total_feature_exec_time = 0
        total_feature_comm_time = 0
        feature_exec_count = 0

        # Get first message from Manager
        status = MPI.Status()
        new_features = comm.recv(source=manager_rank, status=status)
        manager_tag = status.Get_tag()

        # Exit if there are no more tasks
        if manager_tag == MPI_TAGS.MANAGER_FINISH.value:
            break

        # Run jobs until there are not any
        print(f"[petro-dist][w{rank}] new iteration")
        while (manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value):

            t1 = time.time()
            print(f'[petro-dist][w{rank}] executing {len(new_features)} '\
                   'features in parallel')
            with concurrent.futures.ProcessPoolExecutor() as executor:
                future = [
                    executor.submit(petro2.single_feature_run, cur_df,
                                    features_df, f,
                                    parallel_settings['cpu_thrds'])
                    for f in new_features
                ]
                print('all submitted===============')
            t2 = time.time()
            print(f'[petro-dist][w{rank}] ran {len(new_features)} '\
                  f'features in parallel in {t2-t1} secs')

            # # Run all features concurrently
            # for new_feature in new_features:

            #     rmse, mae = petro2.single_feature_run(
            #         cur_df, features_df, new_feature,
            #         parallel_settings['cpu_thrds'])

            # Return results to manager
            results = [f.result() for f in future]
            results = [rmse for (rmse, mae) in results]
            comm.send((list(zip(new_features,
                                results)), parallel_settings['n_cpus']),
                      dest=manager_rank)

            # Wait for new job
            new_feature = comm.recv(source=manager_rank, status=status)
            manager_tag = status.Get_tag()

            t3 = time.time()
            total_feature_exec_time = total_feature_exec_time + (t2 - t1)
            total_feature_comm_time = total_feature_comm_time + (t3 - t2)
            feature_exec_count = feature_exec_count + 1

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_f_set.append(new_best_feature)
        cur_df.loc[:,
                   petro2.f2str(new_best_feature)] = petro2.get_feature_col2(
                       cur_df.index, new_best_feature, features_df)

        t4 = time.time()

        print(f'[petro-dist][w{rank}][profiling] it_full_time: {t4-t0}')
        print(f'[petro-dist][w{rank}][profiling] total_exec_time: '\
              f'{total_feature_exec_time}')
        print(f'[petro-dist][w{rank}][profiling] total_comm_time: '\
              f'{total_feature_comm_time}')
        print(f'[petro-dist][w{rank}][profiling] n_tasks: '\
              f'{feature_exec_count}')

    # Get broadcasted resulting features and errors
    best_result = comm.bcast(None, root=manager_rank)
    return best_result


def main(initial_iteration: int, num_iterations: int, parallel_settings):

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
    seismic_features_names = [
        "FAR",
        "MID",
        # "NEAR_azimuth_",
        # "NEAR_contour-curvature_",
        # "NEAR_curvedness_",
        # "NEAR_dip-angle_",
        # "NEAR_dip-curvature_",
        # "NEAR_envelope_",
        # "NEAR_gaussian-curvature_",
        # "NEAR_gersztenkorn_3-3-11",
        # "NEAR_gersztenkorn_3-3-7",
        # "NEAR_gersztenkorn_3-3-9",
        # "NEAR_gersztenkorn_5-5-11",
        # "NEAR_gersztenkorn_5-5-7",
        # "NEAR_gersztenkorn_5-5-9",
        # "NEAR_gst_3-3-11",
        # "NEAR_gst_3-3-7",
        # "NEAR_gst_3-3-9",
        # "NEAR_gst_5-5-11",
        # "NEAR_gst_5-5-7",
        # "NEAR_gst_5-5-9",
        # "NEAR_instantaneous-frequency_",
        # "NEAR_max-curvature_",
        # "NEAR_mean-curvature_",
        # "NEAR_min-curvature_",
        # "NEAR_most-negative-curvature_",
        # "NEAR_most-positive-curvature_",
        # "NEAR",
        # "NEAR_rms-5_",
        # "NEAR_shape-index_",
        # "NEAR_sobel_5-5-11",
        # "UFAR",
    ]

    # Features which do not need to be expanded on the window
    other_features_names = []

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
        [main_df['x'], main_df['y'], main_df['z']],
        names=common.MAIN_DF_INDEX_NAMES)
    main_df.set_index(index, inplace=True)
    main_df.sort_index(inplace=True)

    index = pd.MultiIndex.from_arrays([xs_np, ys_np, zs_np],
                                      names=common.MAIN_DF_INDEX_NAMES)

    canal_df.set_index(index, inplace=True)
    canal_df.sort_index(inplace=True)

    # Free indexes np arrays
    xs = None
    ys = None
    zs = None
    full_canal_np = None

    # Generate seismic features names
    window = 3
    all_features = other_features_names
    for f in seismic_features_names:
        for i in range(-window, window + 1):
            for j in range(-window, window + 1):
                for k in range(-window, window + 1):
                    all_features.append((f, i, j, k))
    t2 = time.time()

    if initial_iteration > 0:
        main_df = pd.read_csv(f'./tmp_data/predicted{initial_iteration}.csv')
        index = pd.MultiIndex.from_arrays(
            [main_df['x'], main_df['y'], main_df['z']],
            names=common.MAIN_DF_INDEX_NAMES)
        main_df.set_index(index, inplace=True)
        main_df.sort_index(inplace=True)

    print(f'[main] Initial data loading time: {t2-t1}')

    print("[main] Main DataFrame [initial]:")
    print(main_df)

    maxIteration = initial_iteration + num_iterations
    for it in range(initial_iteration, maxIteration):
        t1 = time.time()

        print(f"[main][{it}] Expanding points")
        main_df = expand2.gen_expanded_points(main_df, canal_df, real_wells,
                                              it)
        main_df.sort_index(inplace=True)
        main_df.to_csv(f'tmp_data/expanded{it}.csv', index=False)
        print(main_df)

        t2 = time.time()

        print(f"[main][{it}] Performing feature selection")
        # Only uses real, previously predicted and expanded canal points
        # for feature selection
        feature_selection_points_df = main_df[main_df['real'] != 3]
        print('[main] Points for feature selection:')
        print(feature_selection_points_df)
        if mpi_size == 1:
            best_features_set, best_error = petro2.get_features_sets(
                feature_selection_points_df, features_df, all_features,
                parallel_settings, 10, 4)
        else:
            best_features_set, best_error = petro_dist.get_features_sets(
                feature_selection_points_df, features_df, all_features,
                parallel_settings, 10, 0)

        print(f'[main][{it}] Best features set:'\
              f' {best_features_set} with {best_error} error'
        )

        t3 = time.time()

        print(f"[main][{it}] Performing predictions on new expanded points")
        main_df = apply4.perf_predition(best_features_set, main_df,
                                        features_df)
        print(main_df)
        # main_df.sort_index(inplace=True)
        main_df.to_csv(f'tmp_data/predicted{it}.csv',
                       index=True,
                       index_label=common.MAIN_DF_INDEX_NAMES)

        t4 = time.time()
        print(f'[main][times][{it}] total_it_time {t4-t1}')
        print(f'[main][times][{it}] expansion {t2-t1}')
        print(f'[main][times][{it}] feature_selection {t3-t2}')
        print(f'[main][times][{it}] propagation {t4-t3}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument('--it',
                        dest='initial_it',
                        action='store',
                        default=0,
                        help='Initial iteration (default: 0)')
    parser.add_argument('--nits',
                        dest='num_its',
                        action='store',
                        default=10,
                        help='Number of iterations to run (default: 10)')
    parser.add_argument('--gpu',
                        dest='n_gpus',
                        action='store',
                        default=0,
                        help='Number of GPUs to be used (default: 0)')
    parser.add_argument('--gput',
                        dest='gpu_thrds',
                        action='store',
                        default=0,
                        help='Number of threads to be executed '\
                             'per GPU (default: 0)')
    parser.add_argument('--cpu',
                        dest='n_cpus',
                        action='store',
                        default=1,
                        help='Number of LGB execution threads to be '\
                             'executed by node. Parallel multithreading per '\
                             'LGB thread can be enabled (default: 1)')
    parser.add_argument('--cput',
                        dest='cpu_thrds',
                        action='store',
                        default=1,
                        help='Number of CPU threads to be used '\
                             'per LGB execution (default: 1)')

    args = parser.parse_args()
    parallel_settings = {
        'n_cpus': int(args.n_cpus),
        'cpu_thrds': int(args.cpu_thrds),
        'n_gpus': int(args.n_gpus),
        'gpu_thrds': int(args.gpu_thrds),
    }
    main(int(args.initial_it), int(args.num_its), parallel_settings)
