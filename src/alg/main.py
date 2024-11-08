import argparse
from mpi4py import MPI
import sys
from time import sleep

import config_parser
import mpi_module

import manager
import worker

# Used only for retrieving the shape of a feature
from feature_data.backends.FeatureDataBase import FeatureDataBase

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def config_arg_parser():
    parser = argparse.ArgumentParser(description="Modelagem de "
                                     "Aprendizado Invertido")

    parser.add_argument(
        '--config',
        dest='config_file',
        action='store',
        required=True,
        type=str,
        help="The yaml config file path to be read",
    )

    parser.add_argument(
        '--it',
        dest='load_it',
        action='store',
        required=False,
        default=1,
        type=int,
        help="Iteration to start. E.g., if value is 3, it is assumed "
        "that propagation went through iteration 2. First iteration is "
        "1 (default=1)",
    )

    parser.add_argument(
        '--nits',
        dest='num_its',
        action='store',
        required=False,
        type=int,
        default=1,
        help="Number of iterations to run (default=1).",
    )

    parser.add_argument(
        '--nf',
        dest='num_features',
        action='store',
        required=False,
        type=int,
        default=0,
        help="Number of total features (default=0, i.e., all)",
    )

    parser.add_argument(
        '--nsf',
        dest='num_select_features',
        action='store',
        required=False,
        default=10,
        type=int,
        help="Number of maximum features to be selected (default=10)",
    )

    parser.add_argument(
        '--ntf',
        dest='num_tested_features',
        action='store',
        default=0,
        required=False,
        type=int,
        help="Number of features to be tested before choosing "
        "a selected feature (default=0, i.e., all).",
    )

    parser.add_argument(
        '--wp',
        dest='with_progress',
        action='store_true',
        default=True,
        help="Enable showing progress of iterations. "
        "This can mess the slurm output up.",
    )

    parser.add_argument(
        '--no-wp',
        dest='with_progress',
        action='store_false',
        help="Disables showing progress of iterations.",
    )

    parser.add_argument(
        '-w',
        dest='window',
        action='store',
        required=False,
        type=int,
        help="Size of the displacement window. This value is for one side "
        "only. I.e., a window of 3 would result in a minicube of "
        "7x7x7, with intervals between [-3,3].",
    )

    parser.add_argument(
        '--no-abort',
        dest='no_abort',
        action='store_true',
        default=False,
        required=False,
        help="Disables MPI_ABORT whenever there is an error. "
        "Useful for debugging",
    )

    parser.add_argument(
        '--fso',
        dest='feature_sel_only',
        action='store_true',
        default=False,
        required=False,
        help="Feature selection only. Skip the propagation step. "
        "Useful for performance testing since no changes on the "
        "porosity files are done.",
    )

    parser.add_argument(
        '--fsched-loc',
        dest='fsched_loc',
        action='store_true',
        default=False,
        required=False,
        help="Enable feature locality-aware scheduling.",
    )

    parser.add_argument(
        '--sw',
        dest='small_window',
        action='store_true',
        default=False,
        required=False,
        help="Profiling option. When enabled, only coordinate k (depth) "
        "will have displacement applied to it. This allows for having small "
        "number of displacements per feature. Window size restrictions still "
        "apply, i.e., enabling small_window and using -w 5 on a dataset with "
        "window of 3 will break the application.",
    )

    parser.add_argument(
        '--f-inmem',
        dest='is_feature_in_mem',
        action='store_true',
        default=False,
        required=False,
        help="Enable in-memory storage of features. If enabled, all features "
        "are pre-fetched once before the execution of any iteration or trial.",
    )

    parser.add_argument(
        '--f-cache',
        dest='is_feature_cache',
        action='store_true',
        default=False,
        required=False,
        help="Enable in-memory caching of features. If enabled, entire "
        "features are cached on first access. LRU cache with "
        "configurable cache_lines size.",
    )

    parser.add_argument(
        '--p-dfs',
        dest='is_porosity_dfs',
        action='store_true',
        default=False,
        required=False,
        help="Signals that porosity data is on DFS, meaning that only "
        "a single worker should propagate data. Since the propagated "
        "data is globally available through the DFS, this is ok. The "
        "alternative is to perform propagation on node-local porosity "
        "files, one worker per node.",
    )

    parser.add_argument(
        '--t-shd',
        dest='is_shared_trial_data',
        action='store_true',
        default=False,
        required=False,
        help="Enables shared memory storage of trial data. Shared memory "
        "is only for processes within the same node. Each node has a single "
        "shared trial data structure on memory.",
    )

    parser.add_argument(
        '--t-h5',
        dest='is_h5_trial_data',
        action='store_true',
        default=False,
        required=False,
        help="Enables H5 storage of trial data.",
    )

    parser.add_argument(
        '--t-h5-shd',
        dest='is_h5_shared_trial_data',
        action='store_true',
        default=False,
        required=False,
        help="Enables shared H5 storage of trial data. Shared H5 file "
        "is only for processes within the same node. Each node has a single "
        "shared trial data file.",
    )

    parser.add_argument(
        '--pov',
        dest='pov_canal_path',
        action='store',
        default=None,
        required=False,
        type=str,
        help="Configures propagation and validation for POV. Input path "
        "is from canal data. Canal data should be an h5 file. This file "
        "is opened in read-only mode.",
    )

    parser.add_argument(
        '--pov-no-prop',
        dest='pov_should_prop',
        action='store_false',
        default=True,
        required=False,
        help="Only usable with --pov. When propagating, estimated porosity "
        "will only be used for test. Afterward, original canal points are "
        "reloaded int the H5 porosity structure.",
    )

    parser.add_argument(
        '--tr-chunk',
        dest='training_chunks',
        action='store',
        default=1,
        required=False,
        help="Number of training chunks for incremental learning "
        "(default=1).",
    )

    parser.add_argument(
        '--tr-chunk-seq',
        dest='training_chunks_seq',
        action='store',
        default=None,
        required=False,
        help="Number of training chunks for incremental learning "
        "using sequential chunking. This overwrites --tr-chunk (default=1).",
    )

    return parser


def update_config_file_params_with_args(config: config_parser.Config,
                                        args) -> config_parser.Config:

    config.alg['it'] = int(args.load_it)
    # Tirar esse assert ou mudar para >=0
    assert config.alg['it'] >= 0, f"First iteration is 0, "\
                                 f"but received --it {config.alg['it']}"
    config.alg['num_its'] = int(args.num_its)
    assert config.alg['num_its'] > 0, f"At least 1 iteration should be run, "\
                                 f"but received --nits {config.alg['num_its']}"

    config.alg['max_num_features'] = int(args.num_select_features)
    config.add_param('num_features', int(args.num_features))

    if args.window is not None:
        config.alg['window'] = int(args.window)

    if args.with_progress is not None:
        config.add_param('with_progress', args.with_progress)

    if args.fsched_loc is not None:
        config.add_param('fsched_loc', args.fsched_loc)

    config.add_param('n_training_chunks', int(args.training_chunks))
    if args.training_chunks_seq is not None:
        config.add_param('sequential_chunking', True)
        config.add_param('n_training_chunks', int(args.training_chunks_seq))

    config.add_param('full_depth_chunks', True)

    config.add_param('max_feats_for_trial', int(args.num_tested_features))

    # config.add_param('is_sampling', bool(args.is_sampling))

    config.add_param('feature_sel_only', args.feature_sel_only)
    config.add_param('small_window', args.small_window)
    config.add_param('is_feature_in_mem', args.is_feature_in_mem)
    config.add_param('is_feature_cache', args.is_feature_cache)
    config.add_param('is_porosity_dfs', args.is_porosity_dfs)
    config.add_param('is_shared_trial_data', args.is_shared_trial_data)
    config.add_param('is_h5_trial_data', args.is_h5_trial_data)
    config.add_param('is_h5_shared_trial_data', args.is_h5_shared_trial_data)
    
    # POV exec
    config.add_param('pov_canal_path', args.pov_canal_path)
    config.add_param('pov_should_prop', args.pov_should_prop)

    # Profiling
    #config.add_param('prof_trial_prep_porosity', True)
    # config.add_param('prof_trial_update_feature', True)
    # config.add_param('prof_feature_sel', True)
    # config.add_param('prof_TD_get_values', True)

    # Debug info
    config.add_param('fsched_debug', True)

    return config


def main(args_str=None):
    # Retrieve CLI arguments
    if args_str is None:
        args = config_arg_parser().parse_args()
    else:
        args = config_arg_parser().parse_args(args_str.split(' '))

    # Retrieve config file parameters
    config = config_parser.YAMLConfig(args.config_file)

    # Overwrite config file parameters with CLI args when necessary
    update_config_file_params_with_args(config, args)

    # Initialize the MPI environment, if it's being used.
    # Initialized values and other objects are inserted into config
    mpi_module.initialize(config)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    manager_rank = mpi_size - 1

    # Call MPI.abort() on all processes if one of them breaks
    # This avoids lingering executions after any error occurs
    def _end():
        # This sleep timer allows processes to output error messages
        sleep(3)
        MPI.COMM_WORLD.Abort()

    if not args.no_abort:
        sys.excepthook = lambda x, y, z: _end()

    assert mpi_size > 1, "Two or more processes required to run "\
                         "(mpirun -np 2 python3 main.py)."

    # Load the shape of the first feature into config. All features
    # should have the same shape. This is kind of hacky. Maybe improve this in
    # the future.
    f_paths = [str(p) for p in config.features_files_paths if '.h5' in str(p)]
    f_paths += [str(p) for p in config.features_files_paths if '.npy' in str(p)]
    first_feature_path = str(f_paths[0])
    feature_shape = FeatureDataBase.get_shape(
        first_feature_path, config.get_param('mpi_local_comm'))
    config.add_param('feature_shape', feature_shape)

    if rank == manager_rank:
        try:
            manager.run(config)
        except Exception as e:
            print(f"[manager][main] Exception detected on main:\n{e}")
            raise e
            # print(f"[manager][configs]{config}")
    else:
        try:
            worker.run(config)
        except Exception as e:
            print(f"[worker-{rank}][main] Exception detected on main:\n{e}")
            raise e


if __name__ == "__main__":
    main()
