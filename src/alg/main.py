from __future__ import annotations
import argparse
from collections import namedtuple
import h5py
from mpi4py import MPI
import os
import pathlib
import time
from typing import Callable
from tqdm import tqdm

import apply5_hdf5
import common
import expand4_hdf5
import hdf5_util
import petro5_hdf5
import petro_dist4_hdf5

import config_parser

RunningProcess = namedtuple('RunningProcess',
                            'should_update print_progress_func rank comm')

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


# Return a split communicator for processes within the same node.
# For instance, running 4 nodes and 8 processes would yield 4 different
# communicators with 2 processes.
# This is required for using distributed hdf5. The MPI_File_open routine
# needs the same file for the input MPI communicator, and for running locally
# the files although having the same name and path, are different.
def get_local_node_comm():
    # Get all node names
    local_host_name = MPI.Get_processor_name()
    node_names = comm.allgather(local_host_name)

    # Get unique sorted
    node_names = list(set(node_names))
    node_names.sort()

    # Perform split
    local_color = node_names.index(local_host_name)
    local_comm = comm.Split(local_color)

    # print(f'[main][get_local_node_comm] r{rank}color{local_color}')

    return local_comm

# TODO:
# Refactor to remove this global variable
# Currently, main.py is too complicated and overcrowded to make a simple
# straightforward solution
local_comm = get_local_node_comm()

def print_manager(string):
    """
    Print function for only the manager process
    """
    if rank == manager_rank:
        print(string)


def print_progress(with_progress, r):
    if with_progress:
        return tqdm(r)
    else:
        return r

def mpi_perform_ordered(func):
    """
    Perform 'func' one rank at a time
    """
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
    ret = False
    assigned_nodes = []
    for r in range(mpi_size):
        # Get node name of rank r
        if r == rank:
            cur_node = MPI.Get_processor_name()
            comm.bcast(cur_node, root=r)
        else:
            cur_node = comm.bcast(None, root=r)

        # Update list of seen nodes
        if cur_node not in assigned_nodes:
            assigned_nodes.append(cur_node)

            # If this is the first unique rank of a node, it should update
            if r == rank:
                print(f'rank {r} should update on node {cur_node}')
                ret = True

    return ret


def remove_old_lock_files(lock_file_str: str):
    mpi_perform_ordered(lambda: os.remove(lock_file_str)
                        if os.path.exists(lock_file_str) else None)


def create_lock_file_locally_if_there_isnt_one(lock_file_str: str):
    ret = mpi_perform_ordered(lambda: open(lock_file_str, 'w').close()
                              if not os.path.exists(lock_file_str) else False)

    return ret


def get_print_progress_func(should_update: bool,
                            with_progress: bool) -> Callable:
    """
    Progress printing only enabled for updating process
    """
    if should_update:
        print(f'[main] Rank {rank} is updating h5 file')
        return lambda r: print_progress(with_progress, r)

    return lambda r: r


def get_running_process(with_progress: bool) -> RunningProcess:
    """
    Assign a single process per node to update the local h5 file
    """
    should_update = should_update_local()
    pp = get_print_progress_func(should_update, with_progress)

    return RunningProcess(should_update, pp, rank, comm)


class FeaturesDataLoaderH5():

    def __init__(self, config: config_parser.Config):
        num_features = config.get_param('num_features')
        #This is ok because num_features is either an int or None. The latter works as [:]
        self.features_names = config.features_files_names[:num_features]
        self.features_file_paths = config.features_files_paths[:num_features]

        self.features_files_dict_h5 = {}
        self.features_dict_h5 = {}

    def __enter__(self) -> FeaturesDataLoaderH5:
        for file_path, feat_name in zip(self.features_file_paths,
                                        self.features_names):
            print(f'[main] loading file {file_path}')
            self.features_files_dict_h5[feat_name] = h5py.File(
                file_path, 'r', driver='mpio', comm=local_comm)
            self.features_dict_h5[feat_name] = self.features_files_dict_h5[
                feat_name]['f']

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for file in self.features_files_dict_h5.keys():
            self.features_files_dict_h5[file].close()


class PorosityCubeDataLoaderH5():

    def __init__(self, cube_path: pathlib.Path):
        self.cube_path = pathlib.Path(cube_path)
        self.porosity_data = None
        self.porosity_cube_file = None

    def __enter__(self) -> PorosityCubeDataLoaderH5:
        # For MPI_FILE_OPEN, used by hdf5 with mpi, all files must be opened
        # with the same access/mode: existing file with write permission
        # However, only one process updates this porosity_data_h5 structure
        write_str = 'r+'

        # Real wells' data into a main dataframe
        print("[main] Loading wells values")
        self.porosity_cube_file = h5py.File(self.cube_path,
                                            write_str,
                                            driver='mpio',
                                            comm=local_comm)
        self.porosity_data = self.porosity_cube_file['p']

        all_points = self.porosity_data.size
        self._print_hypercube_stats(all_points)
        self._print_real_well_point_stats(all_points)
        self._print_canal_points_stats(all_points)

        return self

    def _print_hypercube_stats(self, all_points):
        hypercube_shape = self.porosity_data.shape
        print(f'[main] hypercube_shape: {hypercube_shape}')
        print(f'[main] hypercube size: {all_points}')

    def _print_real_well_point_stats(self, all_points: int):
        real_points = hdf5_util.fold_h5_all_clusters(
            self.porosity_data,
            lambda d: len(d[d['real'] == common.RealValues.real]), 0)

        print(f'[main] real well points: {real_points}/{all_points} '\
            f'({(real_points/all_points):%})')

    def _print_canal_points_stats(self, all_points):
        canal_points = hdf5_util.fold_h5_all_clusters(
            self.porosity_data,
            lambda d: len(d[d['real'] == common.RealValues.canal]), 0)

        print(f'[main] canal points: {canal_points}/{all_points} '\
            f'({(canal_points/all_points):.2%})')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.porosity_cube_file.close()


class Algorithm():

    def __init__(self, config: config_parser.Config):
        self.config = config

    def run(self):
        t1 = time.time()

        print("[main] Loading seismic data")
        with FeaturesDataLoaderH5(local_comm, self.config) as feat_dl:
            print("[main] Loading wells values")
            with PorosityCubeDataLoaderH5(
                    self.config.alg['starting_porosity_cube_path'],
                    local_comm) as porosity_dl:
                features_dict_h5 = feat_dl.features_dict_h5
                porosity_data_h5 = porosity_dl.porosity_data

                t2 = time.time()
                print(f'[main] Initial data loading time: {t2-t1}')

                self._run_alg(porosity_data_h5, features_dict_h5)

    def _run_alg(self, porosity_data_h5, features_dict_h5: dict):
        my_process = self.config.get_param('my_process')
        max_iteration = self.config.alg['starting_it'] + self.config.alg[
            'num_its'] + 1
        starting_it = self.config.alg['starting_it'] + 1

        window = self.config.get_param('window')
        window_sizes = self._get_window_sizes(window)
        displacement_cube_shape = self._get_displacement_cube_shape()

        all_features = self._generate_seismic_features_names()

        for it in range(starting_it, max_iteration):
            it_str = f'[it{it}]'

            t1 = time.time()

            self._print_empty_points(porosity_data_h5)

            self._expand_points(porosity_data_h5, it, it_str)

            print(porosity_data_h5)

            t2 = time.time()

            best_features_set = self._feature_selection(
                porosity_data_h5, features_dict_h5, all_features, window_sizes,
                displacement_cube_shape, it_str)

            t3 = time.time()

            print(
                f"[main]{it_str} Performing predictions on new expanded points"
            )
            if my_process.should_update:
                apply5_hdf5.perf_predition(best_features_set, porosity_data_h5,
                                           features_dict_h5, window_sizes,
                                           displacement_cube_shape)

            t4 = time.time()

    def _get_window_sizes(self, window: int) -> tuple:
        return (window, window, window)

    def _get_displacement_cube_shape(self) -> tuple:
        window = self.config.get_param('window')
        return (window * 2 + 1, window * 2 + 1, window * 2 + 1)

    def _generate_seismic_features_names(self) -> list:
        window = self.config.get_param('window')
        all_features = list(self.config.get_param('other_feat_names'))

        num_features = self.config.get_param("num_features")
        #This is ok because num_features is either an int or None. For the latter, it works as [:]
        files_names = self.config.features_files_names[:num_features]

        for f in files_names:
            for i in range(-window, window + 1):
                for j in range(-window, window + 1):
                    for k in range(-window, window + 1):
                        all_features.append((f, i, j, k))

        return all_features

    def _print_empty_points(self, porosity_data_h5):
        empty_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[d['real'] == common.RealValues.empty]), 0)
        print(f'[main] empty points: {empty_points}')

    def _expand_points(self, porosity_data_h5, it: int, it_str: int):
        wells_coords = self.config.wells_as_simple_list()
        my_process = self.config.get_param('my_process')
        print(f"[main]{it_str} Expanding points")
        if my_process.should_update:
            hypercube_shape = porosity_data_h5.shape
            expand4_hdf5.gen_expanded_points(porosity_data_h5, hypercube_shape,
                                             wells_coords, it, it_str,
                                             my_process.print_progress_func)
        else:
            my_rank = self.config.get_param('my_process').rank
            print(f'[main]{it_str}[R{my_rank}] waiting points expansion')

        comm.Barrier()

        self._print_expanded_points(porosity_data_h5, it_str)

    def _print_expanded_points(self, porosity_data_h5, it_str: str):
        to_expand = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.expanded)]), 0)
        my_rank = self.config.get_param('my_process').rank
        print(f'[main]{it_str}[R{my_rank}] Expanded points: {to_expand}')

    def _feature_selection(self, porosity_data_h5, features_dict_h5: dict,
                           all_features: list, window_sizes: tuple,
                           displacement_cube_shape: tuple, it_str: str):

        print(f"[main]{it_str} Performing feature selection")
        max_num_features = self.config.alg['max_num_features']

        self._print_feature_selection_points(porosity_data_h5)

        if mpi_size == 1:
            best_features_set, best_error = petro5_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features, window_sizes,
                displacement_cube_shape, it_str, max_num_features, 0)
        else:
            best_features_set, best_error = petro_dist4_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features, window_sizes,
                displacement_cube_shape, it_str, max_num_features, 5)

        print_manager(f'[main]{it_str} Best features set:'\
                f' {best_features_set} with {best_error} error')

        return best_features_set

    def _print_feature_selection_points(self, porosity_data_h5):
        # Only uses real, previously propagated and expanded canal points
        # for feature selection
        f_sel_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.propagated) |
                            (d['real'] == common.RealValues.real)]), 0)
        print(f'[main] Points for feature selection: {f_sel_points}')


def main(config: config_parser.Config):
    # Data structure is composed by:
    #   x,y,z(depth),
    #   well => Well ID (-1 if it's not an original real point.)
    #                   (Has the ID from the original real well)
    #                   (from which it was expanded.           )
    #   real => [3=expanded, to be propagated, 2=expanded canal,
    #            1=propagated, 0=real well point]
    #   phi  => Porosity value

    my_process = get_running_process(config.get_param('with_progress'))
    config.add_param('my_process', my_process)
    #Not used anymore
    config.remove_param('with_progress')

    config.add_param('window', 3)
    # Features which do not need to be expanded on the window
    config.add_param('other_feat_names', [])

    my_alg = Algorithm(config)
    my_alg.run()


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
                        required=False,
                        help='Number of total features')
    parser.add_argument('--nsf',
                        dest='num_select_features',
                        action='store',
                        required=False,
                        help='Number of maximum features to be '\
                            'selected')
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

    parser.add_argument('--local',
                        dest='local_files',
                        action='store_true',
                        help='Read files from main.py root folder.')
    return parser


def update_config_file_params_with_args(config: config_parser.Config,
                                        args) -> config_parser.Config:
    if args.num_select_features is not None:
        config.alg['max_num_features'] = int(args.num_select_features)

    if args.load_it is not None:
        config.alg['starting_it'] = int(args.load_it)

    if args.num_its is not None:
        config.alg['num_its'] = int(args.num_its)

    if args.num_features is not None:
        #int() of None raise an error so we must check before.
        my_config.add_param('num_features', int(args.num_features))

    if args.with_progress is not None:
        my_config.add_param('with_progress', args.with_progress)

    if args.local_files is not None:
        config.alg[
            'starting_porosity_cube_path'] = f"./{pathlib.Path(config.alg['starting_porosity_cube_path']).name}"
        config.features_folder = "./features/"

    return config


if __name__ == '__main__':

    parser = config_arg_parser()
    args = parser.parse_args()

    my_config = config_parser.YAMLConfig(args.config_file)

    my_config = update_config_file_params_with_args(my_config, args)

    main(my_config)
