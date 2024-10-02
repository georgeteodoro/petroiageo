import sys
import numpy as np
import h5py

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import PopulationBasedTraining

import matplotlib.pyplot as plt
import os

import feature_sel
import config_parser
import common
import mpi_module

from TrialDataSharedNumpy import TrialDataSharedNumpy
from feature_data.FeatureDatasetMMapCache import FeatureDatasetMMapCache

def _load_porosity(config):
    # For MPI_FILE_OPEN, used by hdf5 with mpi, all files must be opened
    # with the same access/mode: existing file with write permission
    # However, only one process should update this porosity_data_h5
    # structure.
    write_str = "r+"

    # Setup HDF5 driver configuration
    # mpi_kwargs = {
    #     'driver': 'mpio',
    #     'comm': config.get_param('mpi_local_comm'),
    # }
    mpi_kwargs = {}

    porosity_cube_file = h5py.File(config.starting_porosity_cube_path,
                                   write_str, **mpi_kwargs)
    porosity_cube_dset = porosity_cube_file[common.POROSITY_DSET_NAME]

    return porosity_cube_file, porosity_cube_dset

def initialize_training_data(config, base_features):
    # Loading porosity file
    train_wells_ids = config.train_wells_ids
    porosity_h5_f, porosity_h5_dset = _load_porosity(config)

    # Prepping trial data
    trial_data = TrialDataSharedNumpy(train_wells_ids, porosity_h5_dset,
                                      config)
    trial_data.prepare_porosity(5)

    # Prepping feature data
    all_features = FeatureDatasetMMapCache(config)

    # Add a single feature
    feature0 = all_features.get_feature(base_features[0])
    feature1 = all_features.get_feature(base_features[1])
    trial_data.commit_feature(feature0, (0,0,0))
    trial_data.commit_feature(feature1, (0,1,0))
    trial_data.commit_feature(feature0, (-1,1,0))

    return trial_data

def initialize_tunner(obj_function):
    search_space = {
        'num_leaves': tune.randint(5, 100),
        'min_data_in_leaf': tune.randint(1, 100),
        'max_depth': tune.randint(1, 10000)
    }

    trainable_with_resources = tune.with_resources(obj_function, {"cpu": 1})

    perturbation_interval = 5
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        # metric="mean_accuracy",
        # mode="max",
        metric="rmse",
        mode="min",
        hyperparam_mutations={
            # distribution for resampling
            "lr": tune.uniform(0.0001, 1),
            # allow perturbations within this set of categorical values
            "momentum": [0.8, 0.9, 0.99],
        },
    )

    num_samples = 4

    # tuner = tune.Tuner(
    #     trainable_with_resources,
    #     run_config=train.RunConfig(
    #         stop={'training_iteration': 50},
    #         checkpoint_config=train.CheckpointConfig(
    #             checkpoint_score_attribute="rmse",
    #             num_to_keep=4,
    #         ),
    #     ),
    #     tune_config = tune.TuneConfig(
    #         scheduler = scheduler,
    #         num_samples = num_samples,
    #         max_concurrent_trials=40,
    #     ),
    #     param_space = search_space,
    # )

    tuner = tune.Tuner(
        trainable_with_resources,
        run_config=train.RunConfig(
            stop={'training_iteration': 50},
        ),
        tune_config = tune.TuneConfig(
            search_alg = OptunaSearch(),
            num_samples = num_samples,
            metric = 'rmse',
            mode = 'min',
            max_concurrent_trials=40,
        ),
        param_space = search_space,
    )

    return tuner

def main():
    if len(sys.argv) != 2:
        print("usage: python3 hpo-test.py CONFIG_PATH")
        return

    # Parse config
    config = config_parser.YAMLConfig(sys.argv[1])
    config.alg['max_num_features'] = 4
    base_features = config.features_files_names
    base_features = [f for f in base_features if f != ".gitkeep"]
    mpi_module.initialize(config)
    
    trial_data = initialize_training_data(config, base_features)

    print(feature_sel.test_new_feature(trial_data, config))

    curr_well_id = config.train_wells_ids[0]
    X_val, y_val = trial_data.get_val_values(curr_well_id)
    X_train, y_train = trial_data.get_train_values(curr_well_id, 0)

    def obj_function(trial_conf):
        rmse, mae = feature_sel._full_train(X_train, y_train, X_val, 
            y_val, trial_conf)
        return {"rmse": rmse}

    tuner = initialize_tunner(obj_function)
    results = tuner.fit()
    print(results.get_best_result(metric="rmse", mode="min").config)


    # # Plot the learning curve for the best trial
    # best_result = results.get_best_result(metric="rmse", mode="min")
    # df = best_result.metrics_dataframe
    # print(df.columns.tolist())
    # # Deduplicate, since PBT might introduce duplicate data
    # # df = df.drop_duplicates(subset="training_iteration", keep="last")
    # df.plot("num_samples", "rmse")
    # plt.xlabel("Training Iterations")
    # plt.ylabel("Test rmse")
    # plt.show()




if __name__ == '__main__':
    main()