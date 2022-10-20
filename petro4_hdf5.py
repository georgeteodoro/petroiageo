# import pandas as pd
import numpy as np
import time
# from numba import jit
import h5py
from math import prod

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

import hdf5_util
import common

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


# def eval_bootstrap(df, num_threads=24):
def eval_bootstrap(cur_h5_seq, wells_id, num_threads=24):
    params['num_threads'] = num_threads

    rmse_list = []
    mae_list = []
    for w in wells_id:
        # Create a sequence object for training and validation
        # This custom sequence allows for partial, out-of-core
        # data loading by lgb
        X_train_seq = cur_h5_seq
        X_train_seq.set_lowo_train(w)
        X_val_seq = cur_h5_seq
        X_val_seq.set_lowo_val(w)

        # Setup for the datasets
        y_train_np = X_train_seq.get_y_np()
        y_val_np = X_val_seq.get_y_np()
        lgb_train_dataset = lgb.Dataset(X_train_seq, y_train_np)
        lgb_eval_dataset = lgb.Dataset(X_val_seq, y_val_np)

        # Perform training
        regressor = lgb.train(
            params,
            lgb_train_dataset,
            num_boost_round=100,
            valid_sets=lgb_eval_dataset,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

        # Calculate error metrics
        pred = regressor.predict(X_val)
        rmse = np.sqrt(np.mean((pred - y_val)**2))
        mae = mean_absolute_error(pred, y_val)
        rmse_list.append(rmse)
        mae_list.append(mae)

    # X = df.values
    # y = df[LABEL_COLUMN_NAME].values
    # a = []
    # b = []

    # # Create groups for one-well-out training
    # logo = LeaveOneGroupOut()
    # groups = df['well']
    # logo.get_n_splits(X, y, groups)
    # logo.get_n_splits(groups=groups)
    # for (train, val) in logo.split(X, y, groups):

    #     # Create training dataset
    #     train_df = df.iloc[train]
    #     X_train = train_df.drop([LABEL_COLUMN_NAME] + UNWANTED_COLUMNS,
    #                             axis=1).values
    #     y_train = train_df[LABEL_COLUMN_NAME].values
    #     lgb_train = lgb.Dataset(X_train, y_train)

    #     # Create validation dataset
    #     val_df = df.iloc[val]
    #     val_df = val_df[val_df['real'] == 0]  # Select real values only
    #     X_val = val_df.drop([LABEL_COLUMN_NAME] + UNWANTED_COLUMNS,
    #                         axis=1).values
    #     y_val = val_df[LABEL_COLUMN_NAME].values
    #     lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    #     # Perform training
    #     regressor = lgb.train(
    #         params,
    #         lgb_train,
    #         num_boost_round=100,
    #         valid_sets=lgb_eval,
    #         callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

    #     # Calculate error metrics
    #     pred = regressor.predict(X_val)
    #     rmse = np.sqrt(np.mean((pred - y_val)**2))
    #     mae = mean_absolute_error(pred, y_val)
    #     a.append(rmse)
    #     b.append(mae)

    return np.mean(a), np.mean(b)


# Convert the feature tuple (e.g., ('NEAR', -3, 1, 2)) to
# string (e.g., 'NEAR/-3,1,2')
def f2str(f_tuple):
    if type(f_tuple) is tuple:
        return f'{f_tuple[0]}/{f_tuple[1]},{f_tuple[2]},{f_tuple[3]}'
    else:
        return f_tuple


# @jit(nopython=True)
# def parallel_read(array_np, indexes, f_x, f_y, f_z):
#     ret = np.empty((len(indexes)), dtype=np.float64)

#     ii = 0
#     for i in indexes:
#         x = max(0, min(SEISMIC_MAX_X, i['x'] + f_x))
#         y = max(0, min(SEISMIC_MAX_Y, i['y'] + f_y))
#         z = max(0, min(SEISMIC_MAX_Z, i['z'] + f_z))
#         # array_np is 1D with 3D indexed data
#         coord = x * (SEISMIC_MAX_Y + 1) * (SEISMIC_MAX_Z +
#                                            1) + y * (SEISMIC_MAX_Z + 1) + z
#         ret[ii] = array_np[coord]
#         ii = ii + 1

#     return ret

# # Uses ndarray instead of pandas access
# def get_feature_col2(indexes, feature, features_df):
#     if type(feature) is tuple:
#         sub_features_np = features_df[feature[0]].values

#         # Numba only accepts ndarrays of concrete types (not object)
#         indexes_ndarray = np.array(indexes.values,
#                                    dtype=[('x', '<u4'), ('y', '<u4'),
#                                           ('z', '<u4')])
#         return parallel_read(sub_features_np, indexes_ndarray, feature[1],
#                              feature[2], feature[3])
#     else:
#         print(f'[petro2][WARNING] non-seismic column created: {feature}')
#         return features_df[features_df.index.isin(indexes)][feature].values

# # This version loads the entire numpy ndarray of features to be evaluated.
# # The data loading is done through a chunked hdf5 file
# # def single_feature_run(cur_df, features_df, cur_feature, num_threads):
# def single_feature_run(cur_h5_seq, features_dict_h5, cur_feature, it_str):

#     t1 = time.time()
#     print(f'[single_feature_run]{it_str} begin {t1}')
#     cur_feature_s = f2str(cur_feature)

#     # Add feature column to current DataFrame
#     # A copy of the current rolling DataFrame is done
#     # in order to avoid inserting and removing columns
#     # The current rolling DataFrame is updated after all
#     # features are tested
#     # test_df = cur_df.copy(deep=False)
#     # t2 = time.time()
#     # test_df.loc[:, cur_feature_s] = get_feature_col2(test_df.index,
#     #                                                  cur_feature, features_df)

#     # Copy the current feature values to the current dataset
#     cur_h5_seq.update_last_col(features_dict_h5[cur_feature_s])

#     t3 = time.time()

#     # Test current feature set
#     rmse, mae = eval_bootstrap(cur_h5_seq, num_threads)
#     t4 = time.time()

#     print(f'[petro2][single_feature_run] copy_time {t2-t1}')
#     print(f'[petro2][single_feature_run] add_col_time {t3-t2}')
#     print(f'[petro2][single_feature_run] eval_bootstrap {t4-t3}')

#     print(f'[single_feature_run] end {t4}')

#     return rmse, mae


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(
        porosity_data_h5,
        features_dict_h5,
        all_features,
        # num_threads,
        it_str,
        exp_n_features,
        f_width=0):

    # # Create a shallow copy of main_df for adding new columns
    # # Data from is main_df is only referenced, not copied
    # cur_df = main_df.copy(deep=False)

    # Creates a temporary h5 structure to maintain the porosity
    # and features data
    cur_h5 = h5py.File('cur.h5', 'w')
    cur_chunksize = (100, 100, 100)  # AUTOMATE LATER
    cur_data_type = [('x', np.int64), ('y', np.int64), ('z', np.int64),
                     ('phi', np.float64), ('well_id', np.int64)]
    cur_data_type = cur_data_type + [(f'f{f}', np.float64)
                                     for f in range(exp_n_features)]
    cur_data_type = np.dtype(cur_data_type)
    n_training_points = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5,
        lambda d: len(d[d['real'] != common.RealValues.empty]), 0)
    print(n_training_points)
    cur_h5_dset = cur_h5.create_dataset('c', (n_training_points, ),
                                        dtype=cur_data_type,
                                        chunks=(prod(cur_chunksize), ))

    # Copy porosity data to cur structure
    # Only copy points which will be used for training, i.e., not empty.
    # Deep copy is required since porosity_data_h5 has all points
    # (including empty points) which won't be used for training, and
    # just filtering these out would return a ndarray in-memory structure.
    # This ndarray can be too large to fit in memory.
    prev_end = 0
    for x_c, y_c, z_c in porosity_data_h5.iter_chunks():
        # Get current chunk
        chunk_np = porosity_data_h5[x_c, y_c, z_c]

        # Filter only one type of points
        training_points = chunk_np[chunk_np['real'] != common.RealValues.empty]

        # Append these porosity values to the current dataset
        cur_h5_dset['x', 'y', 'z', 'phi', 'well_id',
                    prev_end:(prev_end +
                              len(training_points))] = training_points[[
                                  'x', 'y', 'z', 'phi', 'well_id'
                              ]]
        prev_end = prev_end + len(training_points)

    # Create sequence object
    cur_h5_seq = hdf5_util.HDFMultiColSequence(cur_h5_dset, ['x', 'y', 'z'])

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

            # rmse, mae = single_feature_run(cur_df, features_df, cur_feature,
            #                                num_threads)

            # "Add" a column to the dataset. This added feature is later
            # replaced by other features to be tested
            cur_h5_seq.update_last_col(features_dict_h5[f2str(cur_feature)])
            rmse, mae = eval_bootstrap(cur_h5_seq, num_threads)

            results.append((cur_f_set + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            t4 = time.time()
            print(f'[petro2]{it_str} Tested feature'\
                  f'{cur_f_set+ [cur_feature]} with error {rmse}')

        # # Update current DataFrame to add best feature of current iteration
        # cur_f_set.append(best_feature)
        # cur_df.loc[:, f2str(best_feature)] = get_feature_col2(
        #     cur_df.index, best_feature, features_df)

        # Append the best column to the dataset. This is permanent, with
        # regards to the current dataset
        cur_h5_seq.set_last_col(features_dict_h5[f2str(best_feature)])

        t5 = time.time()

        # Print iteration statistics
        print(f'[petro2] fullIt time: {t5-t0}')

    return get_best_features_set(results)
