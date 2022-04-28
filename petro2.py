import pandas as pd
import numpy as np
import sys
import os
import random
import time
from io import StringIO
from numba import jit

from memory_profiler import profile

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = 'phi'
UNWANTED_COLUMNS = ['real', 'well']

RANDOM_STATE = 1

params = {
    "max_bin": 128,
    "max_depth": 10,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 20,
    "verbose": -1,
    "min_data": 10,
    "boost_from_average": True,
    "bagging_freq": 1,
    "random_state": RANDOM_STATE,
}

SEISMIC_MAX_X = 433
SEISMIC_MAX_Y = 645
SEISMIC_MAX_Z = 250

# Profiling variables
avg_col_time = 0
avg_eval_time = 0
feature_run_count = 0


def get_best_features_set(features_sets):
    # Sort by second column (id 1)
    features_sets.sort(key=lambda tup: tup[1])
    best_features_set = features_sets[0][0]
    best_error = features_sets[0][1]

    return best_features_set, best_error


def eval_bootstrap(df, num_threads=24):
    # params['num_threads'] = num_threads

    # print(features)
    X = df.values
    y = df[LABEL_COLUMN_NAME].values
    a = []
    b = []

    # Create groups for one-well-out training
    logo = LeaveOneGroupOut()
    groups = df['well']
    logo.get_n_splits(X, y, groups)
    logo.get_n_splits(groups=groups)
    for (train, val) in logo.split(X, y, groups):

        # Create training dataset
        train_df = df.iloc[train]
        X_train = train_df.drop([LABEL_COLUMN_NAME] + UNWANTED_COLUMNS,
                                axis=1).values
        y_train = train_df[LABEL_COLUMN_NAME].values
        lgb_train = lgb.Dataset(X_train, y_train)

        # Create validation dataset
        val_df = df.iloc[val]
        val_df = val_df[val_df['real'] == 0]  # Select real values only
        X_val = val_df.drop([LABEL_COLUMN_NAME] + UNWANTED_COLUMNS,
                            axis=1).values
        y_val = val_df[LABEL_COLUMN_NAME].values
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # Perform training
        regressor = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_eval,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

        # Calculate error metrics
        pred = regressor.predict(X_val)
        rmse = np.sqrt(np.mean((pred - y_val)**2))
        mae = mean_absolute_error(pred, y_val)
        a.append(rmse)
        b.append(mae)

    return np.mean(a), np.mean(b)


# Convert the feature tuple (e.g., ('NEAR', -3, 1, 2)) to
# string (e.g., 'NEAR/-3,1,2')
def f2str(f_tuple):
    if type(f_tuple) is tuple:
        return f'{f_tuple[0]}/{f_tuple[1]},{f_tuple[2]},{f_tuple[3]}'
    else:
        return f_tuple


@jit(nopython=True)
def parallel_read(array_np, indexes, f_x, f_y, f_z):
    ret = np.empty((len(indexes)), dtype=np.float64)

    ii = 0
    for i in indexes:
        x = max(0, min(SEISMIC_MAX_X, i['x'] + f_x))
        y = max(0, min(SEISMIC_MAX_Y, i['y'] + f_y))
        z = max(0, min(SEISMIC_MAX_Z, i['z'] + f_z))
        # array_np is 1D with 3D indexed data
        coord = x * (SEISMIC_MAX_Y + 1) * (SEISMIC_MAX_Z +
                                           1) + y * (SEISMIC_MAX_Z + 1) + z
        ret[ii] = array_np[coord]
        ii = ii + 1

    return ret


# Uses ndarray instead of pandas access
def get_feature_col2(indexes, feature, features_df):
    if type(feature) is tuple:
        sub_features_np = features_df[feature[0]].values

        # Numba only accepts ndarrays of concrete types (not object)
        indexes_ndarray = np.array(indexes.values,
                                   dtype=[('x', '<u4'), ('y', '<u4'),
                                          ('z', '<u4')])
        return parallel_read(sub_features_np, indexes_ndarray, feature[1],
                             feature[2], feature[3])
    else:
        print(f'[petro2][WARNING] non-seismic column created: {feature}')
        return features_df[features_df.index.isin(indexes)][feature].values


def single_feature_run(cur_df, features_df, cur_feature):
    # print(f"[petro] Testing feature {cur_feature}")

    t1 = time.time()
    cur_feature_s = f2str(cur_feature)

    # Add feature column to current DataFrame
    # A copy of the current rolling DataFrame is done
    # in order to avoid inserting and removing columns
    # The current rolling DataFrame is updated after all
    # features are tested
    test_df = cur_df.copy(deep=False)
    t2 = time.time()
    test_df.loc[:, cur_feature_s] = get_feature_col2(test_df.index,
                                                     cur_feature, features_df)

    t3 = time.time()

    # Test current feature set
    rmse, mae = eval_bootstrap(test_df)
    t4 = time.time()

    print(f'[petro2][single_feature_run] copy_time {t2-t1}')
    print(f'[petro2][single_feature_run] add_col_time {t3-t2}')
    print(f'[petro2][single_feature_run] eval_bootstrap {t4-t3}')

    # global avg_col_time
    # global avg_eval_time
    # global feature_run_count
    # avg_col_time = avg_col_time + (t2 - t1)
    # avg_eval_time = avg_eval_time + (t3 - t2)
    # feature_run_count = feature_run_count + 1

    return rmse, mae


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
# @profile
def get_features_sets(main_df,
                      features_df,
                      all_features,
                      exp_n_features,
                      f_width=0):
    # Create a shallow copy of main_df for adding new columns
    # Data from is main_df is only referenced, not copied
    cur_df = main_df.copy(deep=False)

    # Remove rows from cur_df which don't have an original well
    # ID (i.e., well=-1)
    # cur_df = cur_df[cur_df['well'] != -1]

    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']

    # List of features sets and their error metric
    results = []

    # Find a feature set with exp_n_features features
    for _ in range(exp_n_features):

        t0 = time.time()

        # Reset best feature and its error
        best_error = 10000
        best_feature = ()

        # Test each available feature
        ii = 0
        # print(f'[petro] starting iteration with features:')
        # print(cur_f_set)
        for cur_feature in all_features:
            if cur_feature in cur_f_set:
                continue

            # Early termination for debugging
            if f_width != 0 and ii == f_width:
                break
            ii = ii + 1

            rmse, mae = single_feature_run(cur_df, features_df, cur_feature)

            results.append((cur_f_set + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            t4 = time.time()
            # print(f"[petro2] it time for {len(test_df)} rows (total {t4-t1}):")
            # print(f"[petro2]    col select  {t2-t1}")
            # print(f"[petro2]    training    {t3-t2}")
            # print(f"[petro2]    update best {t4-t3}")
            print(f'[petro2] Tested feature'\
                  f'{cur_f_set+ [cur_feature]} with error {rmse}')

        # Update current DataFrame to add best feature of current iteration
        # print(f'[petro] found best feature: {f2str(cur_feature)}')
        cur_f_set.append(best_feature)
        cur_df.loc[:, f2str(best_feature)] = get_feature_col2(
            cur_df.index, best_feature, features_df)
        # print('[petro] current DF:')
        # print(cur_df)

        t5 = time.time()

        # Print iteration statistics
        global avg_col_time
        global avg_eval_time
        global feature_run_count
        print(f'[petro2] fullIt time: {t5-t0}')
        print(f'[petro2] avg_col_time: {avg_col_time/feature_run_count}')
        print(f'[petro2] avg_eval_time: {avg_eval_time/feature_run_count}')
        avg_col_time = 0
        avg_eval_time = 0
        feature_run_count = 0

    return get_best_features_set(results)
