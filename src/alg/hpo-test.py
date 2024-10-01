import sys
import numpy as np
import h5py

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

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
        'max_depth': tune.randint(25, 10000)
    }

    tuner = tune.Tuner(
        obj_function,
        tune_config = tune.TuneConfig(
            search_alg = OptunaSearch(),
            num_samples = 10,
            metric = 'loss',
            mode = 'min',
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

    def obj_function(trial_conf):
        rmse, mae = feature_sel.test_new_feature(trial_data, config, trial_conf)
        return {"score": rmse}

    tuner = initialize_tunner(obj_function)
    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min").config)

    # # Load data.
    # dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")

    # # Split data into train and validation.
    # train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

    # trainer = LightGBMTrainer(
    #     scaling_config=ScalingConfig(
    #         # Number of workers to use for data parallelism.
    #         num_workers=4,
    #         # Whether to use GPU acceleration. Set to True to schedule GPU workers.
    #         use_gpu=False,
    #     ),
    #     label_column="target",
    #     num_boost_round=20,
    #     params={
    #         # LightGBM specific params
    #         "objective": "binary",
    #         "metric": ["binary_logloss", "binary_error"],
    #     },
    #     datasets={"train": train_dataset, "valid": valid_dataset},
    #     # If running in a multi-node cluster, this is where you
    #     # should configure the run's persistent storage that is accessible
    #     # across all worker nodes.
    #     # run_config=ray.train.RunConfig(storage_path="s3://..."),
    # )
    # result = trainer.fit()
    # print(result.metrics)


if __name__ == '__main__':
    main()