import sys
import numpy as np
import h5py
import argparse

# import adddeps  # fix sys.path
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

# import matplotlib.pyplot as plt
# import os

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

# ===================================================================

trial_data = None
config = None

class AppTuner(MeasurementInterface):

    # def __init__(self, trial_data, args):
    #     super(AppTuner, self).__init__(args)
    #     self.trial_data = trial_data


    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            IntegerParameter('num_leaves', 5, 100))
        manipulator.add_parameter(
            IntegerParameter('min_data_in_leaf', 1, 100))
        manipulator.add_parameter(
            IntegerParameter('max_depth', 1, 10000))
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data

        # print(cfg)
        # print(input)
        # print(limit)
        # print(desired_result)
        # 0/0
        rmse, mae = feature_sel.test_new_feature(trial_data, config)


        # gcc_cmd = 'g++ mmm_block.cpp '
        # gcc_cmd += '-DBLOCK_SIZE='+ cfg['blockSize']
        # gcc_cmd += ' -o ./tmp.bin'

        # compile_result = self.call_program(gcc_cmd)
        # assert compile_result['returncode'] == 0

        # run_cmd = './tmp.bin'

        # run_result = self.call_program(run_cmd)
        # assert run_result['returncode'] == 0

        return Result(time=rmse)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print(f"Optimal block size written to "
              f"mmm_final_config.json: {configuration.data}")
        self.manipulator().save_to_file(configuration.data,
                            'mmm_final_config.json')

# ===================================================================


def main():
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    opt_args, unknown = parser.parse_known_args()
    print(opt_args)
    print(unknown)

    # if len(sys.argv) != 2:
    #     print("usage: python3 hpo-test.py CONFIG_PATH")
    #     return

    if len(unknown) != 1:
        print("usage: python3 hpo-test.py CONFIG_PATH")
        return

    # Parse config
    global config
    config = config_parser.YAMLConfig(unknown[0])
    config.alg['max_num_features'] = 4
    base_features = config.features_files_names
    base_features = [f for f in base_features if f != ".gitkeep"]
    mpi_module.initialize(config)
    
    global trial_data
    trial_data = initialize_training_data(config, base_features)

    print(feature_sel.test_new_feature(trial_data, config))

    AppTuner.main(opt_args)


    # curr_well_id = config.train_wells_ids[0]
    # X_val, y_val = trial_data.get_val_values(curr_well_id)
    # X_train, y_train = trial_data.get_train_values(curr_well_id, 0)

    # def obj_function(trial_conf):
    #     rmse, mae = feature_sel._full_train(X_train, y_train, X_val, 
    #         y_val, trial_conf)
    #     return {"rmse": rmse}

    # tuner = initialize_tunner(obj_function)
    # results = tuner.fit()
    # print(results.get_best_result(metric="rmse", mode="min").config)


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