import pandas as pd
import numpy as np
import time
from mpi4py import MPI
import argparse
from math import prod
import h5py
from tqdm import tqdm
import os

import common
import hdf5_util
import expand4_hdf5
import petro5_hdf5
import petro_dist4_hdf5
import apply5_hdf5

import config_parser

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


# Perform 'func' one rank at a time
def mpi_perform_ordered(func):
    to_update_rank = 0
    while True:
        if rank == to_update_rank:
            to_update_rank = to_update_rank + 1
            ret = func()
            for r in range(to_update_rank, mpi_size):
                comm.send(to_update_rank, r)
            return ret
        else:
            to_update_rank = comm.recv(source=to_update_rank)


# Check whether the current process should update the local h5 files
# Only one process per node should do this
# Although multiple updates works on h5, it is inefficient
def should_update_local():
    lock_file_str = '.h5rank.loc'
    # Remove old files
    mpi_perform_ordered(lambda: os.remove(lock_file_str)
                        if os.path.exists(lock_file_str) else None)
    comm.Barrier()

    # Create lock file locally, if there aren't any
    # mpi_perform_ordered(lambda: open(lock_file_str, 'w').close()
    ret = mpi_perform_ordered(lambda: open(lock_file_str, 'w').close()
                              if not os.path.exists(lock_file_str) else False)

    return ret == None

def get_window_sizes(window:int) -> tuple:
    return (window, window, window)

def get_displacement_cube_shape(window:int) -> tuple:
    return (window * 2 + 1, window * 2 + 1, window * 2 + 1)

def load_data(config:config_parser.Config,
              other_features_names:list, window:int):
    print_manager("[main] Loading seismic data")
    features_files_dict_h5 = {}
    features_dict_h5 = {}
    for f in config.features_files_paths:
        print(f'[main] loading file {f}')
        features_files_dict_h5[f] = h5py.File(f,
                                              'r',
                                              driver='mpio',
                                              comm=comm)
        features_dict_h5[f] = features_files_dict_h5[f]['f']

    # For MPI_FILE_OPEN, used by hdf5 with mpi, all files must be opened
    # with the same access/mode: existing file with write permission
    # However, only one process updates this porosity_data_h5 structure
    write_str = 'r+'

    # Real wells' data into a main dataframe
    print_manager("[main] Loading wells values")
    porosity_data_h5_f = h5py.File(config.alg['starting_porosity_cube_path'],
                                 write_str,
                                 driver='mpio',
                                 comm=comm)
    porosity_data_h5 = porosity_data_h5_f['p']
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
    all_features = other_features_names
    for f in config.features_files_names:
        for i in range(-window, window + 1):
            for j in range(-window, window + 1):
                for k in range(-window, window + 1):
                    all_features.append((f, i, j, k))
    
    return porosity_data_h5, hypercube_shape, features_dict_h5, all_features, porosity_data_h5_f

def main(config:config_parser.Config, num_features: int, parallel_settings, with_progress: bool):
    # Data structure is composed by:
    #   x,y,z(depth),
    #   well => Well ID (-1 if it's not an original real point.)
    #                   (Has the ID from the original real well)
    #                   (from which it was expanded.           )
    #   real => [3=expanded, to be propagated, 2=expanded canal,
    #            1=propagated, 0=real well point]
    #   phi  => Porosity value

    # Assign a single process per node to update the local h5 file
    should_update = should_update_local()
    if should_update:
        print(f'[main] Rank {rank} is updating h5 file')

    # # Progress printing only enabled for updating process
    # if rank == manager_rank:
    if should_update:
        pp = lambda r: print_progress(with_progress, r)
    else:
        pp = lambda r: r

    # Read seismic data and add it to a dataframe
    seismic_features_names = config.features_files_names
    seismic_features_names = seismic_features_names[:num_features]

    # # Features which do not need to be expanded on the window
    other_features_names = []

    t1 = time.time()
    
    window = 3
    
    porosity_data_h5, hypercube_shape, features_dict_h5, all_features, porosity_data_h5_f = load_data(config,
                                                                    other_features_names, window)

    t2 = time.time()
    print(f'[main] Initial data loading time: {t2-t1}')

    displacement_cube_shape = get_displacement_cube_shape(window)
    window_sizes = get_window_sizes(window)

    max_iteration = config.alg['starting_it'] + config.alg['num_its'] + 1
    for it in range(config.alg['starting_it'] + 1, max_iteration):
        it_str = f'[it{it}]'
        it_str_manager = ""
        if rank == manager_rank:
            it_str_manager = it_str

        t1 = time.time()

        empty_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[d['real'] == common.RealValues.empty]), 0)
        print(f'[main] empty points: {empty_points}')

        print(f"[main]{it_str} Expanding points")
        if should_update:
            expand4_hdf5.gen_expanded_points(porosity_data_h5, hypercube_shape,
                                             real_wells, it, it_str, pp)
        else:
            print(f'[main]{it_str}[R{rank}] waiting points expansion')
        comm.Barrier()

        to_expand = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.expanded)]), 0)
        print(f'[main]{it_str}[R{rank}] Expanded points: {to_expand}')

        print(porosity_data_h5)

        t2 = time.time()

        print(f"[main]{it_str} Performing feature selection")

        # Only uses real, previously propagated and expanded canal points
        # for feature selection
        f_sel_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.propagated) |
                            (d['real'] == common.RealValues.real)]), 0)
        print(f'[main] Points for feature selection: {f_sel_points}')

        # tmp = porosity_data_h5
        # print(tmp[(tmp['real'] == common.RealValues.canal_expanded) |
        #            (tmp['real'] == common.RealValues.propagated) |
        #            (tmp['real'] == common.RealValues.real)])

        if mpi_size == 1:
            best_features_set, best_error = petro5_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features, window_sizes,
                displacement_cube_shape, it_str, num_select_features, 1)
        else:
            best_features_set, best_error = petro_dist4_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features, window_sizes,
                displacement_cube_shape, it_str, num_select_features, 10)

        print_manager(f'[main]{it_str} Best features set:'\
              f' {best_features_set} with {best_error} error')

        t3 = time.time()

        print(f"[main]{it_str} Performing predictions on new expanded points")
        if should_update:
            apply5_hdf5.perf_predition(best_features_set, porosity_data_h5,
                                       features_dict_h5, window_sizes,
                                       displacement_cube_shape)

        t4 = time.time()
        print(f'[main][times]{it_str} total_it_time {t4-t1}')
        print(f'[main][times]{it_str} expansion {t2-t1}')
        print(f'[main][times]{it_str} feature_selection {t3-t2}')
        print(f'[main][times]{it_str} propagation {t4-t3}')

    # Close all hdf5 files
    porosity_data_h5_f.close()
    for f in seismic_features_names:
        features_files_dict_h5[f].close()

def config_arg_parser():
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument('--config',
                        dest='config_file',
                        action='store',
                        required=True,
                        help="The yaml config file path to be read")
    parser.add_argument('--it',
                        dest='load_it',
                        action='store',
                        required=False,
                        help='Iteration to load')
    parser.add_argument('--nits',
                        dest='num_its',
                        action='store',
                        required=False,
                        help='Number of iterations to run')
    parser.add_argument('--nf',
                        dest='num_features',
                        action='store',
                        default=10,
                        help='Number of total features')
    parser.add_argument('--nsf',
                        dest='num_select_features',
                        action='store',
                        required=False,
                        help='Number of maximum features to be '\
                            'selected')
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

    return parser
    
def update_config_file_params_with_args(config:config_parser.Config, args) -> config_parser.Config:
    if args.num_features is not None:
        config.alg['max_num_features'] = int(args.num_features)
    
    if args.load_it is not None:
        config.alg['starting_it'] = int(args.load_it)
    
    if args.num_its is not None:
        config.alg['num_its'] = int(args.num_its)
    
    return config

if __name__ == '__main__':

    parser = config_arg_parser()
    args = parser.parse_args()
    parallel_settings = {
        'n_cpus': int(args.n_cpus),
        'cpu_thrds': int(args.cpu_thrds),
        # 'n_gpus': int(args.n_gpus),
        # 'gpu_thrds': int(args.gpu_thrds),
    }

    # import cProfile
    # cProfile.runctx('main(int(args.load_it), int(args.num_its), '\
    #                 'int(args.num_features), int(args.num_select_features), '\
    #                 'parallel_settings, args.with_progress)',
    #                 globals(), locals())
    
    my_config = config_parser.YAMLConfig(args.config_file)

    my_config = update_config_file_params_with_args(my_config, args)

    main(my_config, int(args.num_features), parallel_settings, args.with_progress)