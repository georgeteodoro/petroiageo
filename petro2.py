import pandas as pd
import numpy as np
import sys
import os
import random
import time
from io import StringIO

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
    params['num_threads'] = num_threads

    # print(features)
    # X = df[features].values
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
    return f'{f_tuple[0]}/{f_tuple[1]},{f_tuple[2]},{f_tuple[3]}'


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


# Add the data of cur_feature from features_df to cur_df in-place
def add_feature_col(cur_df, features_df, cur_feature):
    if type(cur_feature) is tuple:
        # If f is a tuple, then the feature is seismic
        cur_f_name = f2str(cur_feature)
        print(f'[petro] creating feature col {cur_f_name}')
        cur_df[cur_f_name] = 0  # New column created
        cur_df.loc[:,
                   cur_f_name] = cur_df.apply(transfer_seismic_feature,
                                              axis=1,
                                              args=(features_df, cur_feature))
    else:
        # If cur_feature is not a tuple, then the feature other, and
        # doesn't need any fancy assignment due to its index
        cur_df.join(features_df[cur_feature], on=['x', 'y', 'z'])


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

    print("[petro] Starting features set search")

    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']

    # List of features sets and their error metric
    results = []

    # Find a feature set with exp_n_features features
    remaining_features = all_features.copy()
    for _ in range(exp_n_features):

        # Reset best feature and its error
        best_error = 10000
        best_feature = ()

        # Test each available feature
        ii = 0
        print(f'[petro] starting iteration with features:')
        print(cur_f_set)
        for cur_feature in remaining_features:

            # Early termination for debugging
            if f_width != 0 and ii == f_width:
                break
            ii = ii + 1

            t1 = time.time()

            # Add feature column to current DataFrame
            # A copy of the current rolling DataFrame is done
            # in order to avoid inserting and removing columns
            # The current rolling DataFrame is updated after all
            # features are tested
            test_df = cur_df.copy(deep=False)
            add_feature_col(test_df, features_df, cur_feature)

            t2 = time.time()
            print(f"col setup time {t2-t1}")

            # Test current feature set
            rmse, mae = eval_bootstrap(test_df)
            results.append((cur_f_set + [cur_feature], rmse, mae))
            t3 = time.time()
            print(f"iter time {t3-t1}")

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

        # Update current DataFrame to add best feature of current iteration
        print(f'[petro] found best feature: {f2str(cur_feature)}')
        cur_f_set.append(best_feature)
        add_feature_col(cur_df, features_df, best_feature)
        print('[petro] current DF:')
        print(cur_df)

    return results

    # i = 0
    # results = []
    # for f1 in all_features:
    #     if i == exp_n_features: break
    #     if f1 in f: continue
    #     k = 1000
    #     x = f1
    #     i = i + 1
    #     j = 0
    #     for f2 in all_features:
    #         if f2 in f: continue
    #         j = j + 1
    #         if f_width != 0 and j == f_width: break
    #         f.append(f2)
    #         print(f)

    #         t1 = time.time()

    #         # Add feature column to current DataFrame
    #         # A copy of the current rolling DataFrame is done
    #         # in order to avoid inserting and removing columns
    #         # The current rolling DataFrame is updated after all
    #         # features are tested
    #         ev_df = cur_df.copy(deep=False)
    #         if type(f2) is tuple:
    #             # If f is a tuple, then the feature is seismic
    #             print('creating feature col')
    #             ev_df[f2str(f2)] = 0  # New column created
    #             # transfer_seismic_feature(ev_df, features_df, f2)
    #             ev_df.loc[:, f2str(f2)] = ev_df.apply(transfer_seismic_feature,
    #                                                   axis=1,
    #                                                   args=(features_df, f2))
    #         else:
    #             # If f2 is not a tuple, then the feature other, and doesn't need
    #             # any fancy assignment due to its index
    #             ev_df.join(features_df[f2], on=['x', 'y', 'z'])

    #         t2 = time.time()
    #         print(f"col setup time {t2-t1}")

    #         A, B = eval_bootstrap(ev_df)
    #         t3 = time.time()
    #         print(f"iter time {t3-t1}")
    #         # s.write(f"{f},{A},{B}")
    #         results.append([f, A, B])
    #         z = A
    #         f.remove(f2)
    #         sys.stdout.flush()
    #         if z < k:
    #             x = f2
    #             k = z
    #     f.append(x)
