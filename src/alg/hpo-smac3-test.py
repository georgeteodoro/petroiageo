import sys
import numpy as np
import h5py
import argparse
import ast
from mpi4py import MPI
from time import time

import feature_sel
import config_parser
import common
import mpi_module

from TrialDataSharedNumpy import TrialDataSharedNumpy
from feature_data.FeatureDatasetMMapCache import FeatureDatasetMMapCache

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as HPInt
from ConfigSpace.hyperparameters import UniformFloatHyperparameter as HPFloat
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

def initialize_training_data(config, it, sel_features):
    # Loading porosity file
    train_wells_ids = config.train_wells_ids
    porosity_h5_f, porosity_h5_dset = _load_porosity(config)

    # Prepping trial data
    trial_data = TrialDataSharedNumpy(train_wells_ids, porosity_h5_dset,
                                      config)

    trial_data.prepare_porosity(it)

    # # Add features
    # if rank == manager_rank:

    # Prepping feature data
    all_features = FeatureDatasetMMapCache(config)
    if rank == 0:
        print(all_features)
        print(sel_features)

    for i, (f_name, disp) in enumerate(sel_features):
        if rank == 0:
            print(f'adding [{i}/{len(sel_features)}] {f_name}: {disp}')
        f = all_features.get_feature(f_name)
        trial_data.commit_feature(f, disp)

    return trial_data

# ===================================================================

trial_data = None
config = None

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

def worker(trial_data, config):
    while True:
        trial = comm.recv()

        # Break case
        if trial is None:
            return
        
        #print(f"[worker] got {dict(trial.config)}")

        t0 = time()
        rmse, mae = feature_sel.test_new_feature(trial_data, 
            config, hyperparams=dict(trial.config))
        t1 = time()

        #print(f"[worker] run result: {rmse}")

        comm.send((trial, t1-t0, rmse), dest=manager_rank)

    #print("[worker] done...")



def main():
    parser = argparse.ArgumentParser(description="HPO POV")

    parser.add_argument(
        '--config',
        dest='config_file',
        action='store',
        required=True,
        type=str,
        help="The yaml config file path to be read",
    )

    parser.add_argument(
        '--best-f',
        dest='best_features_file',
        action='store',
        required=True,
        type=str,
        help="Best features selection file.",
    )

    parser.add_argument(
        '--it',
        dest='load_it',
        action='store',
        required=False,
        default=20,
        type=int,
        help="Number of rings to load for testing. (default=20)",
    )

    parser.add_argument(
        '--nfs',
        dest='num_features',
        action='store',
        required=False,
        default=10,
        type=int,
        help="Number of features to use for training. (default=10)",
    )

    parser.add_argument(
        '--n-trials',
        dest='num_trials',
        action='store',
        required=False,
        default=25600,
        type=int,
        help="Number of HPO trials to run. (default=25,600)",
    )

    args = parser.parse_args()

    # Features to be used
    it = int(args.load_it)
    with open(args.best_features_file) as f_sets:
        sel_features = ast.literal_eval(f_sets.readlines()[it])
        #sel_features = ast.literal_eval(f_sets.readline())
        if rank == 0:
            print(f"Features: {sel_features}")
    
    # sel_features = [('FAR', (0,1,0)), ('FAR', (1,2,0))]

    # Parse config
    config = config_parser.YAMLConfig(args.config_file)
    config.alg['max_num_features'] = len(sel_features)
    base_features = config.features_files_names
    base_features = [f for f in base_features if f != ".gitkeep"]
    mpi_module.initialize(config)

    # Prepare porosity only on workers
    if rank != manager_rank:
        if rank == 0:
            print(f"Prepping it {it}")
        trial_data = initialize_training_data(config, it, sel_features)
        
    # Run default config a single time and return result to manager
    if rank == 0: 
        print(f"[r{rank}] Running default trial")
        t0 = time()
        default_rmse, _ = feature_sel.test_new_feature(trial_data, config)
        t1 = time()
        print(f"[r{rank}] Done default trial: {default_rmse} in {t1-t0} secs")
        comm.send(default_rmse, dest=manager_rank)
    if rank == manager_rank:
        default_rmse = comm.recv()

    # Run remaining configs
    if rank != manager_rank:
        worker(trial_data, config)
    else:
        smac_config_space = ConfigurationSpace()
        smac_config_space.add([
            HPInt('num_leaves', lower=5, upper=100, default_value=20),
            HPInt('min_data_in_leaf', lower=1, upper=100, default_value=20),
            HPInt('max_depth', lower=1, upper=10000, default_value=10),
            HPInt('max_bin', lower=64, upper=255, default_value=128),
            HPFloat('learning_rate', lower=0.0001, upper=1.0, default_value=0.1),
            HPInt('num_iterations', lower=1, upper=1000, default_value=100),
        ])

        max_trials = args.num_trials
        scenario = Scenario(smac_config_space, deterministic=True, 
                            n_trials=max_trials, n_workers=1)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1,  # We basically use one seed per config only
        )
        smac = HyperparameterOptimizationFacade(scenario, "empty", 
                intensifier=intensifier, overwrite=True)
        # trials = [smac.ask() for _ in range(max_trials)]

        # Do first pass to start all workers
        assert(max_trials >= mpi_size-2)
        for r in range(mpi_size-1):
            trial = smac.ask()
            #print(f"[manager] sending {dict(trial.config)} to w{r}")
            comm.send(trial, dest=r)

        # Do remaining trials
        best_rmse = float('inf')
        best_conf = None
        status = MPI.Status()
        for _ in range(max_trials - mpi_size + 1):
            # Get trial results
            (trial, trial_time, rmse) = comm.recv(status=status)
            print(f"[manager] got trial {dict(trial.config)} "
                  f"with rmse {rmse:.6f} in {trial_time:.2f} secs")
            smac.tell(trial, TrialValue(cost=rmse, time=trial_time), save=True)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_conf = dict(trial.config)

            # Send next trial
            new_trial = smac.ask()
            #print(f"[manager] sending new {dict(new_trial.config)}")
            comm.send(new_trial, dest=status.Get_source())

        # Finish all remaining workers
        for _ in range(mpi_size-1):
            (trial, trial_time, rmse) = comm.recv(status=status)
            smac.tell(trial, TrialValue(cost=rmse, time=trial_time), save=True)
            print(f"[manager] ending, got trial {dict(trial.config)} "
                  f"with rmse {rmse:.6f} in {trial_time:.2f} secs")

            if rmse < best_rmse:
                best_rmse = rmse
                best_conf = dict(trial.config)

            #print(f"[manager] sending none to {status.Get_source()}")
            comm.send(None, dest=status.Get_source())


        # Parallel smac.optimize() uses dask and pickles the train function.
        # We use h5py and mpi object which cannot be pickled, thus no parallelism
        # incumbent = smac.optimize()

        print(smac.runhistory.get_configs())

        print(f"Best error: {best_rmse}")
        print(f"best config: {best_conf}")
        print(f"improved by {default_rmse - best_rmse:.6f} - "
              f"{default_rmse/best_rmse:.2f}x")

    # MPI.Finalize()
    return

if __name__ == '__main__':
    main()
