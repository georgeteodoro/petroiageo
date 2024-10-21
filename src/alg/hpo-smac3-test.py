import sys
import numpy as np
import h5py
import argparse
import ast

import feature_sel
import config_parser
import common
import mpi_module

from TrialDataSharedNumpy import TrialDataSharedNumpy
from feature_data.FeatureDatasetMMapCache import FeatureDatasetMMapCache

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as HPInt
from smac.runhistory.dataclasses import TrialValue

################
# Requirements:
# apt install swig
# pip3 install smac dask dask-expr


def _load_porosity(config):
    write_str = "r"
    mpi_kwargs = {}

    porosity_cube_file = h5py.File(config.starting_porosity_cube_path,
                                   write_str, **mpi_kwargs)
    porosity_cube_dset = porosity_cube_file[common.POROSITY_DSET_NAME]

    return porosity_cube_file, porosity_cube_dset

def initialize_training_data(config, sel_features):
    # Loading porosity file
    train_wells_ids = config.train_wells_ids
    porosity_h5_f, porosity_h5_dset = _load_porosity(config)

    # Prepping trial data
    trial_data = TrialDataSharedNumpy(train_wells_ids, porosity_h5_dset,
                                      config)
    trial_data.prepare_porosity(config.alg['num_its']-1)

    # Prepping feature data
    all_features = FeatureDatasetMMapCache(config)
    print(all_features)

    # Add features
    print(sel_features)
    for f_name, disp in sel_features:
        print(f'adding {f_name}: {disp}')
        f = all_features.get_feature(f_name)
        trial_data.commit_feature(f, disp)

    # feature0 = all_features.get_feature(base_features[0])
    # feature1 = all_features.get_feature(base_features[1])
    # trial_data.commit_feature(feature0, (0,0,0))
    # trial_data.commit_feature(feature1, (0,1,0))
    # trial_data.commit_feature(feature0, (-1,1,0))

    return trial_data

# ===================================================================

trial_data = None
config = None

# min, max, default
config_space = {'num_leaves': (5, 100, 20),
                'min_data_in_leaf': (1, 100, 20),
                'max_depth': (1, 10000, 10),}

def main():
    # parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    # opt_args, unknown = parser.parse_known_args()
    # print(opt_args)
    # print(unknown)

    if len(sys.argv) != 3:
        print("usage: python3 hpo-test.py CONFIG_PATH FEATURES_PATH")
        return

    # Features to be used
    with open(sys.argv[2]) as f_sets:
        sel_features = ast.literal_eval(f_sets.readline())
        print(f"Features: {sel_features}")
    
    sel_features = [('FAR', (0,1,0)), ('FAR', (1,2,0))]

    # Parse config
    global config
    config = config_parser.YAMLConfig(sys.argv[1])
    config.alg['max_num_features'] = len(sel_features)
    base_features = config.features_files_names
    base_features = [f for f in base_features if f != ".gitkeep"]
    mpi_module.initialize(config)


    print(f"Prepping it {config.alg['num_its']}")
    global trial_data
    trial_data = initialize_training_data(config, sel_features)

    print(f"Running single trial")
    default_rmse = feature_sel.test_new_feature(trial_data, config)[0]
    print()

    def train(trial_hyperparams: Configuration, seed: int = 0) -> float:
        rmse, mae = feature_sel.test_new_feature(trial_data, 
            config, hyperparams=trial_hyperparams)
        return rmse

    smac_config_space = ConfigurationSpace()
    smac_config_space.add([
        HPInt('num_leaves', lower=5, upper=100, default_value=20),
        HPInt('min_data_in_leaf', lower=1, upper=100, default_value=20),
        HPInt('max_depth', lower=1, upper=10000, default_value=10),
    ])

    max_trials = 3
    scenario = Scenario(smac_config_space, deterministic=False, 
                        n_trials=max_trials, n_workers=1)
    smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
    trials = [smac.ask() for _ in range(max_trials)]

    for trial in trials:
        print(dict(trial.config))
        rmse, mae = feature_sel.test_new_feature(trial_data, 
                config, hyperparams=dict(trial.config))
        print(rmse)
        smac.tell(trial, TrialValue(cost=rmse), save=True)

    # Parallel smac.optimize() uses dask and pickles the train function.
    # We use h5py and mpi object which cannot be pickled, thus no parallelism
    # incumbent = smac.optimize()

    # # Get cost of default configuration
    # default_cost = smac.validate(None)
    # print(f"Default cost: {default_cost}")

    print(smac.runhistory.get_configs())

    # Let's calculate the cost of the incumbent
    incumbent = smac.optimize() # No more trials, just get the results
    best_rmse = smac.validate(incumbent)
    print(f"Best error: {best_rmse}")
    print(f"best config: {dict(incumbent)}")
    print(f"improved by {default_rmse - best_rmse:.6f} - "
          f"{default_rmse/best_rmse:.2f}x")

if __name__ == '__main__':
    main()