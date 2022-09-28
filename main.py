#import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from mpi4py import MPI, rc
import argparse
import math
from dask.dataframe import read_parquet
from dask.distributed import performance_report

import dask_utils
import common
import expand3
import petro3
import petro_dist2
import apply4

# Constants
# hypercube_shape = (434, 646, 251)
real_wells = [(134, 227), (146, 500), (167, 186), (174, 365), (200, 102),
              (236, 113), (250, 315), (287, 242), (230, 194), (344, 276)]

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

print(f'=== MPI threadsafeness level: {rc.thread_level}')


def print_manager(string):
    if rank == manager_rank:
        print(string)


def main(load_iteration: int, num_iterations: int, parallel_settings):
    # Instantiate pandas dataframe for all data
    # Data structure is composed by:
    #   x,y,z(depth),
    #   well => Well ID (-1 if it's not an original real point.)
    #                   (Has the ID from the original real well)
    #                   (from which it was expanded.           )
    #   real => [3=expanded, to be propagated, 2=expanded canal,
    #            1=propagated, 0=real well point]
    #   phi  => Porosity value

    # Read seismic data and add it to a dataframe
    seismic_features_names = [
        "FAR",
        "MID",
        "NEAR_azimuth_",
        "NEAR_contour-curvature_",
        "NEAR_curvedness_",
        "NEAR_dip-angle_",
        "NEAR_dip-curvature_",
        "NEAR_envelope_",
        "NEAR_gaussian-curvature_",
        "NEAR_gersztenkorn_3-3-11",
        "NEAR_gersztenkorn_3-3-7",
        "NEAR_gersztenkorn_3-3-9",
        "NEAR_gersztenkorn_5-5-11",
        "NEAR_gersztenkorn_5-5-7",
        "NEAR_gersztenkorn_5-5-9",
        "NEAR_gst_3-3-11",
        "NEAR_gst_3-3-7",
        "NEAR_gst_3-3-9",
        "NEAR_gst_5-5-11",
        "NEAR_gst_5-5-7",
        "NEAR_gst_5-5-9",
        "NEAR_instantaneous-frequency_",
        "NEAR_max-curvature_",
        "NEAR_mean-curvature_",
        "NEAR_min-curvature_",
        "NEAR_most-negative-curvature_",
        "NEAR_most-positive-curvature_",
        "NEAR",
        "NEAR_rms-5_",
        "NEAR_shape-index_",
        "NEAR_sobel_5-5-11",
        "UFAR",
    ]
    seismic_features_names = seismic_features_names[:2]

    # Features which do not need to be expanded on the window
    other_features_names = []

    t1 = time.time()
    print_manager("[main] Loading seismic data")
    # features_ddf, hypercube_shape = seismic_data2.get_all_seismic_data(
    #     seismic_features_names + other_features_names)
    hypercube_shape = np.load(f'./dados/{seismic_features_names[0]}.npy').shape
    features_ddf = read_parquet('dados/seismic_features.parquet',
                                calculate_divisions=True)

    # For the whole dask indexing to work the chunksize must be set:
    # All partitions have dask_chunksize points, while the last has
    # dask_chunksize points or less.
    # Otherwise, the algorithm may break (seg fault) or run with
    # wrong displacements indices for features rows
    dask_chunksize = len(features_ddf.get_partition(0))

    # print_manager("[main] Features DataFrame:")
    print_manager(f'[main] hypercube_shape: {hypercube_shape}')
    print_manager(f'[main] features_ddf with size {len(features_ddf)}:')
    print_manager(features_ddf)

    # Real wells' data into a main dataframe
    print_manager("[main] Loading wells values")
    # main_ddf, canal_ddf = wells_data2.get_canal_and_real_data(
    main_ddf = read_parquet('dados/initial_main_ddf.parquet',
                            calculate_divisions=True)

    print_manager(f'main_ddf with size {len(main_ddf)}:')
    print_manager(main_ddf)

    all_points = len(main_ddf)
    real_points = len(main_ddf[main_ddf['real'] == common.RealValues.real])
    canal_points = len(main_ddf[main_ddf['real'] == common.RealValues.canal])

    print_manager(f'real well points: {real_points}/{all_points} '\
          f'({(real_points/all_points):%})')

    print_manager(f'canal points: {canal_points}/{all_points} '\
          f'({(canal_points/all_points):.2%})')

    # Generate seismic features names
    window = 3
    all_features = other_features_names
    for f in seismic_features_names:
        for i in range(-window, window + 1):
            for j in range(-window, window + 1):
                for k in range(-window, window + 1):
                    all_features.append((f, i, j, k))

    # Load previous iteration values, if required
    if load_iteration > 0:
        print_manager('TODO LOAD PREVIOUS IT')
        return
        # main_df = pd.read_csv(f'./tmp_data/predicted{load_iteration}.csv')
        # index = pd.MultiIndex.from_arrays(
        #     [main_df['x'], main_df['y'], main_df['z']],
        #     names=common.MAIN_DF_INDEX_NAMES)
        # main_df.set_index(index, inplace=True)
        # main_df.sort_index(inplace=True)

    # Get divisions list to enable correct indexing
    # (which is partition-dependent)
    divisions = main_ddf.divisions

    t2 = time.time()
    print_manager(f'[main] Initial data loading time: {t2-t1}')

    max_iteration = load_iteration + num_iterations + 1
    for it in range(load_iteration + 1, max_iteration):
        t1 = time.time()

        print_manager(f"[main][{it}] Expanding points")
        main_ddf = expand3.gen_expanded_points(main_ddf, hypercube_shape,
                                               real_wells, it)

        expanded_ddf = main_ddf[
            (main_ddf['real'] == common.RealValues.expanded) |
            (main_ddf['real'] == common.RealValues.canal_expanded)]
        print_manager(f'points to expand: {len(expanded_ddf)}')
        print_manager(expanded_ddf)
        # main_ddf.to_csv(f'tmp_data/expanded{it}.csv', index=True)

        t2 = time.time()

        print_manager(f"[main][{it}] Performing feature selection")
        # Only uses real, previously propagated and expanded canal points
        # for feature selection
        feature_selection_points_ddf = main_ddf[
            (main_ddf['real'] == common.RealValues.propagated) |
            (main_ddf['real'] == common.RealValues.canal_expanded) |
            (main_ddf['real'] == common.RealValues.real)]
        feature_selection_points_ddf = feature_selection_points_ddf.repartition(
            divisions=divisions).persist()

        print_manager(f'[main] Points for feature selection with size '\
              f'{len(feature_selection_points_ddf)}: ')
        print_manager(feature_selection_points_ddf)

        # if mpi_size == 1:
        #     best_features_set, best_error = petro3.get_features_sets(
        #         feature_selection_points_ddf, features_ddf, all_features,
        #         parallel_settings, 4, 4)
        # else:
        best_features_set, best_error = petro_dist2.get_features_sets(
            feature_selection_points_ddf, features_ddf, all_features,
            hypercube_shape, dask_chunksize, parallel_settings, 10, 0)

        print_manager(f'[main][{it}] Best features set:'\
              f' {best_features_set} with {best_error} error'
        )

        t3 = time.time()

        return

        print_manager(
            f"[main][{it}] Performing predictions on new expanded points")
        main_df = apply4.perf_predition(best_features_set, main_df,
                                        features_df)
        print_manager(main_df)
        # main_df.sort_index(inplace=True)
        main_df.to_csv(f'tmp_data/predicted{it}.csv',
                       index=True,
                       index_label=common.MAIN_DF_INDEX_NAMES)

        t4 = time.time()
        print_manager(f'[main][times][{it}] total_it_time {t4-t1}')
        print_manager(f'[main][times][{it}] expansion {t2-t1}')
        print_manager(f'[main][times][{it}] feature_selection {t3-t2}')
        print_manager(f'[main][times][{it}] propagation {t4-t3}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument('--it',
                        dest='load_it',
                        action='store',
                        default=0,
                        help='Iteration to load (default: 0=none)')
    parser.add_argument('--nits',
                        dest='num_its',
                        action='store',
                        default=10,
                        help='Number of iterations to run (default: 10)')
    # parser.add_argument('--gpu',
    #                     dest='n_gpus',
    #                     action='store',
    #                     default=0,
    #                     help='Number of GPUs to be used (default: 0)')
    # parser.add_argument('--gput',
    #                     dest='gpu_thrds',
    #                     action='store',
    #                     default=0,
    #                     help='Number of threads to be executed '\
    #                          'per GPU (default: 0)')
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
        # 'n_gpus': int(args.n_gpus),
        # 'gpu_thrds': int(args.gpu_thrds),
    }

    dask_utils.initialize_dask(np=rank, w=1, t=1, mem=20, disk=20)
    with performance_report(filename=f"dask-report-{rank}.html"):
    # if True:
        main(int(args.load_it), int(args.num_its), parallel_settings)
