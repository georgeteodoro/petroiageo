import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from time import time

from TrialDataBase import TrialDataBase
from config_parser import Config
import common

def _train_regr(X_train, y_train, X_val, y_val, hyperparams, regressor=None):
    '''
    Single trial given only the base input data and hyperparams.
    Required for HPO.
    '''

    hp = hyperparams.copy()
    num_iterations = hp.pop('num_iterations')

    lgb_train_dataset = lgb.Dataset(X_train, y_train)
    lgb_eval_dataset = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train_dataset,
    )
    regressor = lgb.train(
        hp,
        lgb_train_dataset,
        init_model=regressor,
        num_boost_round=num_iterations,
        valid_sets=lgb_eval_dataset,
        keep_training_booster=True,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False)
        ],
    )

    return regressor

def _full_train(X_train, y_train, X_val, y_val, hyperparams):
    regressor = _train_regr(X_train, y_train, X_val, y_val, hyperparams)

    pred = regressor.predict(X_val)
    rmse = np.sqrt(np.mean((pred - y_val)**2))
    mae = mean_absolute_error(y_val, pred)

    return rmse, mae

def test_new_feature(trial_data: TrialDataBase, config: Config, hyperparams=None):
    '''
    Trains a model with trial_data, returning the metric values for the
    trained model.
    Abstracts any low-level assumptions about where data is located.
    Out-of-core training through incremental learning is supported, however
    not required. It is possible to configure n_training_chunks=1 to disable
    the incremental learning.
    '''

    trial_hyperparams = common.training_params
    if hyperparams is not None:
        trial_hyperparams.update(hyperparams)

    assert len(
        trial_data) > 0, "[feature_sel][test_new_feature] Empty TrialData"
    profile = config.get_param('prof_feature_sel')

    n_training_chunks = config.get_param('n_training_chunks')
    train_wells_ids = config.train_wells_ids

    # Initialize metrics lists
    rmse_list = []
    mae_list = []

    t1 = time()

    # Leave-one-well-out
    for curr_well_id in train_wells_ids:
        t11 = time()

        # Reset model for incremental learning
        regressor = None

        # Validation data is not chunked, thus it only needs to be
        # retrieved once with all data
        X_val, y_val = trial_data.get_val_values(curr_well_id)
        t12 = time()

        # There are no data from this well on trial data
        if len(X_val) == 0:
            continue

        # Performs incremental learning on all chunks
        for chunk_id in range(n_training_chunks):
            t121 = time()

            # Setup training data
            X_train, y_train = trial_data.get_train_values(
                curr_well_id, chunk_id)

            # X_train=None if there are no validation points available. This can
            # only happen if there is only 1 well being propagated.
            if len(X_train) == 0:
                return None

            t122 = time()

            regressor = _train_regr(X_train, y_train, 
                X_val, y_val, trial_hyperparams, regressor)
            
            t123 = time()

            if profile:
                print(f"[feature_sel] train_size: {len(X_train)}")
                print(f"[feature_sel] val_size: {len(X_val)}")
                
                print(f"[feature_sel] well[{curr_well_id}] "
                      f"chunk[{chunk_id+1}/{n_training_chunks}] "
                      f"prep: {t122-t121:.4f}")
                print(f"[feature_sel] well[{curr_well_id}] "
                      f"chunk[{chunk_id+1}/{n_training_chunks}] "
                      f"train: {t123-t122:.4f}")
                print(f"[feature_sel] well[{curr_well_id}] "
                      f"chunk[{chunk_id+1}/{n_training_chunks}] "
                      f"len: {len(y_train)}")

        t13 = time()
        
        # Calculate error metrics
        pred = regressor.predict(X_val)
        rmse = np.sqrt(np.mean((pred - y_val)**2))
        mae = mean_absolute_error(y_val, pred)
        rmse_list.append(rmse)
        mae_list.append(mae)

        t14 = time()

        if profile:
            print(f"[feature_sel] well[{curr_well_id}] prep_val: {t12-t11}")
            print(f"[feature_sel] well[{curr_well_id}] well_final: {t13-t12}")
            print(f"[feature_sel] well[{curr_well_id}] calc_metrics: {t14-t13}")

    t2 = time()
    if profile:
        print(f"[feature_sel] final_feature_time: {t2-t1}")

    return np.mean(rmse_list), np.mean(mae_list)
