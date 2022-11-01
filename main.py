import pandas as pd
import numpy as np
import time
from mpi4py import MPI
import argparse
from math import prod
import h5py
from tqdm import tqdm

import common
import hdf5_util
import expand4_hdf5
import petro4_hdf5
# import petro_dist2
import apply5_hdf5

# Constants
# hypercube_shape = (434, 646, 251)
real_wells = [(134, 227), (146, 500), (167, 186), (174, 365), (200, 102),
              (236, 113), (250, 315), (287, 242), (230, 194), (344, 276)]

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


# Print function for only the manager process
def print_manager(string):
    if rank == manager_rank:
        print(string)


def print_progress(with_progress, r):
    if with_progress:
        return tqdm(r)
    else:
        return r


def main(load_iteration: int, num_iterations: int, parallel_settings,
         with_progress: bool):
    # Instantiate pandas dataframe for all data
    # Data structure is composed by:
    #   x,y,z(depth),
    #   well => Well ID (-1 if it's not an original real point.)
    #                   (Has the ID from the original real well)
    #                   (from which it was expanded.           )
    #   real => [3=expanded, to be propagated, 2=expanded canal,
    #            1=propagated, 0=real well point]
    #   phi  => Porosity value

    # Progress printing only enabled for manager process
    if rank == manager_rank:
        pp = lambda r: print_progress(with_progress, r)
    else:
        pp = lambda r: r

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
    features_files_dict_h5 = {}
    features_dict_h5 = {}
    for f in seismic_features_names:
        features_files_dict_h5[f] = h5py.File(f'./dados/{f}.h5', 'r')
        features_dict_h5[f] = features_files_dict_h5[f]['f']

    # Real wells' data into a main dataframe
    print_manager("[main] Loading wells values")
    porosity_data_h5 = h5py.File(f'./dados/porosity_data.h5', 'r+')['p']
    hypercube_shape = porosity_data_h5.shape
    all_points = porosity_data_h5.size

    real_points = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5,
        lambda d: len(d[d['real'] == common.RealValues.real]), 0)
    canal_points = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5,
        lambda d: len(d[d['real'] == common.RealValues.canal]), 0)

    print_manager(f'[main] hypercube_shape: {hypercube_shape}')
    print_manager(f'[main] hypercube size: {all_points}')

    print_manager(f'[main] real well points: {real_points}/{all_points} '\
          f'({(real_points/all_points):%})')

    print_manager(f'[main] canal points: {canal_points}/{all_points} '\
          f'({(canal_points/all_points):.2%})')

    # Generate seismic features names
    window = 3
    displacement_cube_shape = (window * 2 + 1, window * 2 + 1, window * 2 + 1)
    all_features = other_features_names
    for f in seismic_features_names:
        for i in range(-window, window + 1):
            for j in range(-window, window + 1):
                for k in range(-window, window + 1):
                    all_features.append((f, i, j, k))

    # # Load previous iteration values, if required
    # if load_iteration > 0:
    #     print_manager('TODO LOAD PREVIOUS IT')
    #     return
    #     # main_df = pd.read_csv(f'./tmp_data/predicted{load_iteration}.csv')
    #     # index = pd.MultiIndex.from_arrays(
    #     #     [main_df['x'], main_df['y'], main_df['z']],
    #     #     names=common.MAIN_DF_INDEX_NAMES)
    #     # main_df.set_index(index, inplace=True)
    #     # main_df.sort_index(inplace=True)

    # # Get divisions list to enable correct indexing
    # # (which is partition-dependent)
    # divisions = main_ddf.divisions

    t2 = time.time()
    print_manager(f'[main] Initial data loading time: {t2-t1}')

    max_iteration = load_iteration + num_iterations + 1
    for it in range(load_iteration + 1, max_iteration):
        it_str = f'[it{it}]'
        it_str_manager = ""
        if rank == manager_rank:
            it_str_manager = it_str

        t1 = time.time()

        empty_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[d['real'] == common.RealValues.empty]), 0)
        print_manager(f'[main] empty points: {empty_points}')

        print_manager(f"[main]{it_str} Expanding points")
        expand4_hdf5.gen_expanded_points(porosity_data_h5, hypercube_shape,
                                         real_wells, it, it_str_manager, pp)

        to_expand = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.expanded)]), 0)
        print_manager(f'[main] Expanded points: {to_expand}')

        print(porosity_data_h5)

        t2 = time.time()

        print_manager(f"[main]{it_str} Performing feature selection")

        # Only uses real, previously propagated and expanded canal points
        # for feature selection
        f_sel_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.propagated) |
                            (d['real'] == common.RealValues.real)]), 0)
        print_manager(f'[main] Points for feature selection: {f_sel_points}')

        if mpi_size == 1:
            best_features_set, best_error = petro4_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features,
                displacement_cube_shape, it_str, 1, 1)
        else:
            best_features_set, best_error = petro_dist3_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features,
                hypercube_shape, 10, 0)

        print_manager(f'[main]{it_str} Best features set:'\
              f' {best_features_set} with {best_error} error')

        t3 = time.time()

        print_manager(
            f"[main]{it_str} Performing predictions on new expanded points")
        apply5_hdf5.perf_predition(best_features_set, porosity_data_h5,
                                   features_dict_h5, displacement_cube_shape)
        # print_manager(main_df)
        # main_df.sort_index(inplace=True)
        # main_df.to_csv(f'tmp_data/predicted{it}.csv',
        #                index=True,
        #                index_label=common.MAIN_DF_INDEX_NAMES)

        t4 = time.time()
        print_manager(f'[main][times]{it_str} total_it_time {t4-t1}')
        print_manager(f'[main][times]{it_str} expansion {t2-t1}')
        print_manager(f'[main][times]{it_str} feature_selection {t3-t2}')
        print_manager(f'[main][times]{it_str} propagation {t4-t3}')


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

    parser.add_argument('--wp',
                        dest='with_progress',
                        action='store_true',
                        default=True,
                        help='Enable showing progress of iterations. '\
                             'This can mess the slurm output up.')
    parser.add_argument('--no-wp',
                        dest='with_progress',
                        action='store_false',
                        help='Disables showing progress of iterations.')

    args = parser.parse_args()
    parallel_settings = {
        'n_cpus': int(args.n_cpus),
        'cpu_thrds': int(args.cpu_thrds),
        # 'n_gpus': int(args.n_gpus),
        # 'gpu_thrds': int(args.gpu_thrds),
    }

    main(int(args.load_it), int(args.num_its), parallel_settings,
         args.with_progress)
