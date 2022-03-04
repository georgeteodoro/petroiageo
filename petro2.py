import pandas as pd
import numpy as np
import sys
import os
import random
import time
from io import StringIO
from numba import jit

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = 'phi'
# UNWANTED_COLUMNS = ['real', 'well', 'rho', 'vs', 'vp']
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
    "random_state": 0,
}

SEISMIC_MAX_X = 433
SEISMIC_MAX_Y = 645
SEISMIC_MAX_Z = 250


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


# Transfer a seismic feature value with a displaced coordinate
# to the main DataFrame
# To be used by DataFrame.apply
def transfer_seismic_feature(ds, features_df, f):
    # Bound x,y,z coordinates
    x = max(0, min(SEISMIC_MAX_X, ds.name[0] + f[1]))
    y = max(0, min(SEISMIC_MAX_Y, ds.name[1] + f[2]))
    z = max(0, min(SEISMIC_MAX_Z, ds.name[2] + f[3]))

    # Return seismic value for given coordinate
    return features_df.loc[(x, y, z), f[0]]


def get_feature_col(indexes, feature, features_df):
    if type(feature) is tuple:
        ret = np.empty(len(indexes))
        sub_features = features_df[feature[0]]
        ii = 0
        print(f'[get_feature_col] index len: {len(indexes)}')
        for i in indexes:
            x = max(0, min(SEISMIC_MAX_X, i[0] + feature[1]))
            y = max(0, min(SEISMIC_MAX_Y, i[1] + feature[2]))
            z = max(0, min(SEISMIC_MAX_Z, i[2] + feature[3]))
            ret[ii] = sub_features.loc[(x, y, z)]
            ii = ii + 1
        return ret

    else:
        return features_df[features_df.index.isin(indexes)][feature].values


@jit(nopython=True)
def parallel_read(array_np, indexes, f_x, f_y, f_z):
    ret = np.empty((len(indexes)), dtype=np.float64)

    ii = 0
    for i in indexes:
        x = max(0, min(SEISMIC_MAX_X, i[0] + f_x))
        y = max(0, min(SEISMIC_MAX_Y, i[1] + f_y))
        z = max(0, min(SEISMIC_MAX_Z, i[2] + f_z))
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
                                   dtype=[('x', '<u2'), ('y', '<u2'),
                                          ('z', '<u2')])
        return parallel_read(sub_features_np, indexes_ndarray, feature[1],
                             feature[2], feature[3])
    else:
        return features_df[features_df.index.isin(indexes)][feature].values


# Add the data of cur_feature from features_df to cur_df in-place
def add_feature_col(cur_df, features_df, cur_feature):
    if type(cur_feature) is tuple:
        # If f is a tuple, then the feature is seismic
        cur_f_name = f2str(cur_feature)
        # print(f'[petro] creating feature col {cur_f_name}')
        cur_df[cur_f_name] = 0  # New column created
        cur_df.loc[:,
                   cur_f_name] = cur_df.apply(transfer_seismic_feature,
                                              axis=1,
                                              args=(features_df, cur_feature))
    else:
        # If cur_feature is not a tuple, then the feature other, and
        # doesn't need any fancy assignment due to its index
        cur_df.join(features_df[cur_feature])
        # cur_df = pd.merge(cur_df,
        #          features_df[cur_feature],
        #          left_index=True,
        #          right_index=True,
        #          copy=False)


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features
#   used for debugging
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
    cur_df = cur_df[cur_df['well'] != -1]

    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']
    cur_f_set_s = []

    # List of features sets and their error metric
    results = []

    # Find a feature set with exp_n_features features
    # remaining_features = all_features.copy()
    for _ in range(exp_n_features):

        # Reset best feature and its error
        best_error = 10000
        best_feature = ()

        # Test each available feature
        ii = 0
        # print(f'[petro] starting iteration with features:')
        # print(cur_f_set)
        for cur_feature in all_features:
            print(f"[petro] Testing feature {cur_feature}")

            cur_feature_s = f2str(cur_feature)

            if cur_feature_s in cur_f_set_s:
                continue

            t1 = time.time()
            # Early termination for debugging
            if f_width != 0 and ii == f_width:
                break
            ii = ii + 1

            # print(f"[petro] Testing feature {cur_feature} \
            #     on feature set {cur_f_set}")

            # Add feature column to current DataFrame
            # A copy of the current rolling DataFrame is done
            # in order to avoid inserting and removing columns
            # The current rolling DataFrame is updated after all
            # features are tested
            test_df = cur_df.copy(deep=False)
            # add_feature_col(test_df, features_df, cur_feature)
            test_df.loc[:, cur_feature_s] = get_feature_col2(
                test_df.index, cur_feature, features_df)

            t2 = time.time()

            # Test current feature set
            rmse, mae = eval_bootstrap(test_df)
            results.append((cur_f_set + [cur_feature], rmse, mae))
            t3 = time.time()

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            t4 = time.time()
            print(f"[petro] it time for {len(test_df)} rows (total {t4-t1}):")
            print(f"[petro]    col select  {t2-t1}")
            print(f"[petro]    training    {t3-t2}")
            print(f"[petro]    update best {t4-t3}")

        # Update current DataFrame to add best feature of current iteration
        # print(f'[petro] found best feature: {f2str(cur_feature)}')
        cur_f_set.append(best_feature)
        cur_f_set_s.append(f2str(best_feature))
        # add_feature_col(cur_df, features_df, best_feature)
        cur_df.loc[:, f2str(best_feature)] = get_feature_col2(
            cur_df.index, best_feature, features_df)
        # print('[petro] current DF:')
        # print(cur_df)

    return results
