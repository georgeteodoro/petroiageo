import pandas as pd
import numpy as np
import sys
import os
import random
import time
from io import StringIO
from numba import jit, prange, config
from math import floor
import ctypes
import gc

import dask.dataframe as dd
from dask_ml.metrics import mean_absolute_error

# from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

# used to create an empty array with only the size descriptor
from scipy.sparse import csc_matrix

# Parameters
LABEL_COLUMN_NAME = 'phi'
UNWANTED_COLUMNS = ['well_id', 'real', 'phi_weights']

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
    # "tree_learner": "data",
}

# SEISMIC_MAX_X = 433
# SEISMIC_MAX_Y = 645
# SEISMIC_MAX_Z = 250


def get_best_features_set(features_sets):
    # Sort by second column (id 1)
    features_sets.sort(key=lambda tup: tup[1])
    best_features_set = features_sets[0][0]
    best_error = features_sets[0][1]

    return best_features_set, best_error


def eval_bootstrap(ddf, num_threads=24):
    params['num_threads'] = num_threads
    print('=====================================')

    X = ddf.drop(columns=UNWANTED_COLUMNS)
    y = ddf[LABEL_COLUMN_NAME]

    # Create groups for one-well-out training
    logo = LeaveOneGroupOut()
    groups = ddf['well_id'].compute()

    rmse_list = []
    mae_list = []

    train_shape = (len(X), len(X.columns))
    for (train, val) in logo.split(csc_matrix(train_shape), groups=groups):
        X_train = X.loc[train]
        y_train = y.loc[train]

        X_val = X.loc[val]
        y_val = y.loc[val]

        # Prepare trainer/regressor
        regressor = lgb.DaskLGBMRegressor(
            boosting_type="gbdt",
            n_estimators=100,  # num_boost_round
            # max_bin=128,
            learning_rate=0.1,
            max_depth=10,
            eval_metric="mae",
            num_leaves=20,
            # verbose=-1,
            min_child_samples=10,  # min_data
            # boost_from_average=True,
            subsample_freq=1,  # bagging_freq
            random_state=RANDOM_STATE,
        )

        # Train model
        regressor.fit(
            X_train,
            y_train,
            eval_set=zip(X_val, y_val),
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

        # Perform predictions and evaluate the error
        pred = regressor.predict(X_val)
        mae = mean_absolute_error(pred, y_val)
        mae_list.append(mae)

    return None, np.mean(mae_list)

    # X = df.values
    # y = df[LABEL_COLUMN_NAME].values

    # Create groups for one-well-out training
    # logo = LeaveOneGroupOut()
    # groups = df['well']
    # logo.get_n_splits(X, y, groups)
    # logo.get_n_splits(groups=groups)
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
        rmse_list.append(rmse)
        mae_list.append(mae)

    return np.mean(rmse_list), np.mean(mae_list)


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


# def update_row(row, disloc, ddf, hypercube_shape):
#     (feature, xd, yd, zd) = disloc

#     x = row['x']
#     y = row['y']
#     z = row['z']

#     new_x = min(max(0, x + xd), hypercube_shape[0] - 1)
#     new_y = min(max(0, y + yd), hypercube_shape[1] - 1)
#     new_z = min(max(0, z + zd), hypercube_shape[2] - 1)

#     new_index = new_x * hypercube_shape[2] * hypercube_shape[
#         1] + new_y * hypercube_shape[2] + new_z

#     return ddf.loc[new_index][feature]

# # Wraps the dask DataFrame to avoid it being partitioned by map_partitions
# class SimpleWrapper:

#     def __init__(self, ddf):
#         self.ddf = ddf

# def update_row_partition(df_partition, feature, hypercube_shape,
#                          features_ddf_w):
#     return df_partition.apply(update_row,
#                               axis=1,
#                               args=(feature, features_ddf_w.ddf,
#                                     hypercube_shape))

# # Uses dask to perform assignment
# # def get_feature_col3(indexes, feature, features_ddf, hypercube_shape):
# def get_feature_col3(feature, features_ddf, hypercube_shape):
#     if type(feature) is tuple:
#         return features_ddf[['x', 'y', 'z']].map_partitions(
#             update_row_partition,
#             feature,
#             hypercube_shape,
#             features_ddf_w=SimpleWrapper(features_ddf),
#             meta=(None, float))

#         # sub_features_np = features_df[feature[0]].values

#         # # Numba only accepts ndarrays of concrete types (not object)
#         # indexes_ndarray = np.array(indexes.values,
#         #                            dtype=[('x', '<u4'), ('y', '<u4'),
#         #                                   ('z', '<u4')])
#         # return parallel_read(sub_features_np, indexes_ndarray, feature[1],
#         #                      feature[2], feature[3])
#     else:
#         print(f'[petro2][ERROR] non-seismic column created: {feature}')
#         return -1
#         # return features_df[features_df.index.isin(indexes)][feature].values


# Return indices of rows to be read and the location of which partition
# has the required row
@jit(nopython=True, parallel=True)
def get_loc(part_np, feature, shape, chunksize):
    loc_np = np.empty(len(part_np), dtype=np.int64)
    part_loc_np = np.empty(len(part_np), dtype=np.int64)
    for i in prange(len(part_np)):
        newx = min(max(part_np[i, 0] + feature[1], 0.0), shape[0] - 1.0)
        newy = min(max(part_np[i, 1] + feature[2], 0.0), shape[1] - 1.0)
        newz = min(max(part_np[i, 2] + feature[3], 0.0), shape[2] - 1.0)
        loc_id = newx * shape[2] * shape[1] + newy * shape[2] + newz

        loc_np[i] = loc_id
        part_loc_np[i] = int(loc_id / chunksize)

    return loc_np, part_loc_np


# Returns the numpy array with the displaced rows from a given partition.
# If the row is not present on this current partition, returns the old value.
@jit(nopython=True, parallel=True)
def get_part_col_given_part(part_id_np, part_valid_f_np, feature_part_np,
                            min_f_id, max_f_id, chunksize):
    out_np = np.empty((len(part_id_np)), dtype=np.float64)

    for i in prange(len(part_id_np)):
        if (part_id_np[i] >= min_f_id) & (part_id_np[i] <= max_f_id):
            out_np[i] = feature_part_np[part_id_np[i] % chunksize]
        else:
            out_np[i] = part_valid_f_np[i]

    return out_np


# Workaround to reduce memory usage by dask, thus minimizing memory spillage
def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


# Returns a numpy array of rows for a displaced feature for a single dask
# partition (i.e., pandas DF).
def get_part_col(part, features_w, features_min, features_max, shape, feature,
                 chunksize):
    trim_memory()

    # Early retur for empty partitions
    if len(part) == 0:
        return []

    print(part['well_id'].head())
    loc_np, part_loc_np = get_loc(part.to_numpy(), feature, shape, chunksize)

    # print('loc_np')
    # print(loc_np)

    # Get list of partitions required to lookup data
    part_to_fill = set(part_loc_np)
    del part_loc_np
    gc.collect()
    trim_memory()

    out_np = np.empty((len(part)), dtype=np.float64)
    print(f'len(part): {len(part)}')
    # out_np = np.zeros((len(part)), dtype=np.float64)

    print(f'part_to_fill: {part_to_fill}')

    for p in part_to_fill:
        feature_part = features_w.ddf.get_partition(p)[feature[0]]
        feature_part_np = feature_part.compute().to_numpy()
        out_np = get_part_col_given_part(loc_np, out_np, feature_part_np,
                                         features_min[int(p)],
                                         features_max[int(p)], chunksize)

    print(f'out_np (size: {len(out_np)})')
    print('')
    # print(out_np)
    return out_np


def single_feature_run(cur_ddf, features_ddf, cur_feature, hypercube_shape,
                       dask_chunksize, num_threads):

    t1 = time.time()
    print(f'[single_feature_run] begin {t1}')
    cur_feature_s = f2str(cur_feature)

    # Add feature column to current DataFrame
    # A copy of the current rolling DataFrame is done
    # in order to avoid inserting and removing columns
    # The current rolling DataFrame is updated after all
    # features are tested
    test_ddf = cur_ddf.copy()
    t2 = time.time()
    print(f'getting col {cur_feature}')

    # print(f'test_ddf (size: {len(test_ddf)}):')
    # print(test_ddf)
    # print(test_ddf.head(npartitions=-1))
    # print('cur_ddf:')
    # print(cur_ddf.head(npartitions=-1))

    # test_ddf[cur_feature_s] = get_feature_col3(test_ddf.index, cur_feature,
    #                                            features_ddf, hypercube_shape)
    # test_ddf[cur_feature_s] = get_feature_col3(cur_feature, features_ddf,
    #                                            hypercube_shape)

    # Find the min/max indices of each partition a ahead of time only once
    features_min_index_np = features_ddf.index.map_partitions(
        min).compute().to_numpy()
    features_max_index_np = features_ddf.index.map_partitions(
        max).compute().to_numpy()

    # Wrapper of a dask df to avoid it being transformed into a pandas df
    # This allows ddfs passed to map_partitions to retain partition information
    class Wrapper(object):

        def __init__(self, ddf):
            self.ddf = ddf

    # Run column filter on every dask df partition
    test_ddf[cur_feature_s] = test_ddf.map_partitions(get_part_col,
                                                      Wrapper(features_ddf),
                                                      features_min_index_np,
                                                      features_max_index_np,
                                                      hypercube_shape,
                                                      cur_feature,
                                                      dask_chunksize,
                                                      align_dataframes=False,
                                                      meta=(None, int))

    t3 = time.time()
    test_ddf = test_ddf.persist()
    print(test_ddf.head(npartitions=-1))
    print(f'got col in {t3-t1} secs')

    # Test current feature set
    rmse, mae = eval_bootstrap(test_ddf, num_threads)
    t4 = time.time()

    print(f'[petro2][single_feature_run] copy_time {t2-t1}')
    print(f'[petro2][single_feature_run] add_col_time {t3-t2}')
    print(f'[petro2][single_feature_run] eval_bootstrap {t4-t3}')

    print(f'[single_feature_run] end {t4}')

    return rmse, mae


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(main_df,
                      features_df,
                      all_features,
                      num_threads,
                      exp_n_features,
                      f_width=0):
    # Create a shallow copy of main_df for adding new columns
    # Data from is main_df is only referenced, not copied
    cur_df = main_df.copy(deep=False)

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
        for cur_feature in all_features:
            if cur_feature in cur_f_set:
                continue

            # Early termination for debugging
            if f_width != 0 and ii == f_width:
                break
            ii = ii + 1

            rmse, mae = single_feature_run(cur_df, features_df, cur_feature,
                                           num_threads)

            results.append((cur_f_set + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            t4 = time.time()
            print(f'[petro2] Tested feature'\
                  f'{cur_f_set+ [cur_feature]} with error {rmse}')

        # Update current DataFrame to add best feature of current iteration
        cur_f_set.append(best_feature)
        cur_df.loc[:, f2str(best_feature)] = get_feature_col2(
            cur_df.index, best_feature, features_df)

        t5 = time.time()

        # Print iteration statistics
        print(f'[petro2] fullIt time: {t5-t0}')

    return get_best_features_set(results)
