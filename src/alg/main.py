import argparse
import pathlib
from tqdm import tqdm

import config_parser
import mpi_module
from inverted_learning_interface import BaseInvertedLearning
from h5_porosity_data_loader import H5PorosityDataLoader
from h5_seismic_data_loader import H5SeismicDataLoader
from h5_expand_alg import H5ExpandAlg
from h5_feature_selection_alg import H5FeatureSelectionAlg
from h5_apply_alg import H5ApplyAlg


# DEPRECATED
# Still need to figure out logging
def print_progress(with_progress, r):
    if with_progress:
        return tqdm(r)
    else:
        return r


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
        config.add_param('num_features', int(args.num_features))
    else:
        config.add_param('num_features', 0)

    if args.with_progress is not None:
        config.add_param('with_progress', args.with_progress)

    if (args.local_files is not None) and args.local_files:
        config[
            'starting_porosity_cube_path'] = f"./{pathlib.Path(config['starting_porosity_cube_path']).name}"
        config.features_folder = "./features/"

    config.add_param('full_depth_chunks', True)
    config.add_param('window', 3)
    config.add_param('max_tested_features', 1)

    return config


def main():
    # Retrieve CLI arguments
    args = config_arg_parser().parse_args()

    # Retrieve config file parameters
    config = config_parser.YAMLConfig(args.config_file)

    # Overwrite config file parameters with CLI args when necessary
    update_config_file_params_with_args(config, args)

    # Initialize the MPI environment, if it's being used.
    # Initialized values and other objects are inserted into config
    mpi_module.initialize(config)

    # Run base algorithm
    alg = BaseInvertedLearning(H5SeismicDataLoader(config),
                               H5PorosityDataLoader(config),
                               H5ExpandAlg(config),
                               H5FeatureSelectionAlg(config),
                               H5ApplyAlg(config), config)
    alg.run()


if __name__ == '__main__':
    main()
