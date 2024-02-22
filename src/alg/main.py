import argparse
from mpi4py import MPI
import sys
from time import sleep

import config_parser
import mpi_module

import manager
import worker


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

    return parser


def update_config_file_params_with_args(config: config_parser.Config,
                                        args) -> config_parser.Config:

    config.alg['it'] = int(args.load_it)
    assert config.alg['it'] > 0, f"First iteration is 1, "\
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

    config.add_param('full_depth_chunks', True)

    config.add_param('_max_feats_for_trial', int(args.num_tested_features))

    # config.add_param('is_sampling', bool(args.is_sampling))

    config.add_param('feature_sel_only', args.feature_sel_only)

    # Profiling
    # config.add_param('prof_trial_prep_porosity', True)
    # config.add_param('prof_trial_update_feature', True)
    # config.add_param('prof_feature_sel', True)

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

    if rank == manager_rank:
        manager.run(config)
        # print(f"[manager][configs]{config}")
    else:
        worker.run(config)


if __name__ == "__main__":
    main()
