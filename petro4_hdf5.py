import numpy as np
from time import time
import h5py
from math import prod
from copy import copy

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

import hdf5_util
import common

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


def get_best_features_set(features_sets):
    # Sort by second column (id 1)
    features_sets.sort(key=lambda tup: tup[1])
    best_features_set = features_sets[0][0]
    best_error = features_sets[0][1]

    return best_features_set, best_error


# def eval_bootstrap(df, num_threads=24):
def eval_bootstrap(cur_h5_seq, wells_id, num_threads=24):
    params['num_threads'] = num_threads

    profiling = True

    rmse_list = []
    mae_list = []
    well_id = 0

    # Shallow-copy of the data
    X_train_seq = copy(cur_h5_seq)
    X_val_seq = copy(cur_h5_seq)

    for w in wells_id:
        t0 = time()

        # Create a sequence object for training and validation
        # This custom sequence allows for partial, out-of-core
        # data loading by lgb
        X_train_seq.set_lowo_train(w)
        X_val_seq.set_lowo_val(w)

        # print(f'[eval_bootstrap][w{well_id}] train_size: '\
        #       f'{X_train_seq.__len__()}')
        # print(f'[eval_bootstrap][w{well_id}] val_size: '\
        #       f'{X_val_seq.__len__()}')
        well_id = well_id + 1

        # Setup for the datasets
        y_train_np = X_train_seq.get_y_np()
        y_val_np = X_val_seq.get_y_np()
        lgb_train_dataset = lgb.Dataset(X_train_seq, y_train_np)
        lgb_eval_dataset = lgb.Dataset(X_val_seq, y_val_np)
        t1 = time()
        if profiling:
            print(f'[petro4_hdf5][eval_bootstrap][w{w}] Setup in {t1-t0}')

        # Perform training
        regressor = lgb.train(
            params,
            lgb_train_dataset,
            num_boost_round=100,
            valid_sets=lgb_eval_dataset,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        t2 = time()
        if profiling:
            print(f'[petro4_hdf5][eval_bootstrap][w{w}] Training in {t2-t1}')

        # Calculate error metrics
        pred = regressor.predict(X_val_seq.get_X_np())
        rmse = np.sqrt(np.mean((pred - y_val_np)**2))
        mae = mean_absolute_error(pred, y_val_np)
        rmse_list.append(rmse)
        mae_list.append(mae)
        t3 = time()
        if profiling:
            print(f'[petro4_hdf5][eval_bootstrap][w{w}] Evaluating in {t3-t2}')

    return np.mean(rmse_list), np.mean(mae_list)


def insert_filtered_feature(cur_h5_dset, cur_h5_seq, features_dict_h5,
                            cur_feature, hypercube_shape,
                            displacement_cube_shape):

    profile_time = True

    t0 = time()
    for chunk_slice in cur_h5_dset.iter_chunks():
        t1 = time()
        chunk_np = cur_h5_dset[chunk_slice]
        # print(f'[insert_filtered_feature] cur slice: {chunk_slice}')
        # print(f'[insert_filtered_feature] chunk size: {len(chunk_np)}')

        # Get the coordinates list
        coord_3d_np = chunk_np[['x', 'y', 'z']]

        # Get the shape of the hypercube with a border of minimum and maximum
        # displaced points
        # displaced_hypercube_shape = [x + 1 for x in displacement_cube_shape]
        displaced_hypercube_shape = np.array(hypercube_shape) + (
            np.array(displacement_cube_shape) - 1)

        t2 = time()
        if profile_time:
            print(f'[insert_filtered_feature] get_slice_time: {t2-t1}')

        # Apply the displacement
        coord_planar_np = coord_3d_np.copy()

        for (coord_s, d_id) in [('x', 0), ('y', 1), ('z', 2)]:
            coord_planar_np[coord_s] = coord_planar_np[coord_s] + cur_feature[
                d_id + 1] + ((displacement_cube_shape[d_id] - 1) / 2)

        t3 = time()
        if profile_time:
            print(f'[insert_filtered_feature] appply_disp_time: {t3-t2}')

        # Convert 3d coordinates to planar
        coord_planar_np = coord_planar_np[
            'z'].flat + coord_planar_np['y'].flat * displaced_hypercube_shape[
                2] + coord_planar_np['x'].flat * displaced_hypercube_shape[
                    2] * displaced_hypercube_shape[1]

        coord_planar_np.sort()

        t4 = time()
        if profile_time:
            print(f'[insert_filtered_feature] 3d_2_planar_time: {t4-t3}')

        # We use a generator in order to avoid copying the displaced feature
        # data into a ndarray variable, to later copy it to the cur_h5_seq
        # object.
        def _feature_generator(features_dict_h5, coord_planar_np, f):
            chunk_size = 10
            chunk_len = len(coord_planar_np)
            if len(coord_planar_np) / chunk_size > chunk_size:
                chunk_len = int(len(coord_planar_np) / chunk_size)

                for beg in range(0,
                                 len(coord_planar_np) - chunk_len, chunk_len):
                    chunk_slice = slice(beg, beg + chunk_len)
                    yield chunk_slice, features_dict_h5[f][
                        coord_planar_np[chunk_slice]]
            else:
                yield slice(0, chunk_len), features_dict_h5[f][coord_planar_np]

        feature_gen = _feature_generator(features_dict_h5, coord_planar_np,
                                         cur_feature[0])
        t5 = time()
        if profile_time:
            print(f'[insert_filtered_feature] generator_time: {t5-t4}')

        # Insert the generator displaced feature data into cur_h5_seq
        cur_h5_seq.update_last_col_chunk(feature_gen, coord_planar_np)

        t6 = time()
        if profile_time:
            print(
                f'[insert_filtered_feature] insert_disp_feature_time: {t6-t5}')

    t7 = time()
    if profile_time:
        print(f'[insert_filtered_feature] full_time: {t7-t0}')


def create_tmp_dset(porosity_data_h5,
                    is_training_point_f,
                    n_features,
                    suf_str='',
                    features_only=False):

    profiling = True

    t0 = time()
    # Creates a temporary h5 structure to maintain the porosity
    # and features data
    cur_h5 = h5py.File(f'cur{suf_str}.h5', 'w')
    # cur_chunksize = (100, 100, 100)  # AUTOMATE LATER
    cur_chunksize = (100, 10, 10)  # AUTOMATE LATER
    if features_only:
        cur_data_type = [('x', np.int64), ('y', np.int64), ('z', np.int64),
                         ('phi', np.float64)]
    else:
        cur_data_type = [('x', np.int64), ('y', np.int64), ('z', np.int64),
                         ('phi', np.float64), ('well_id', np.int64)]

    cur_data_type = cur_data_type + [(f'f{f}', np.float64)
                                     for f in range(n_features)]
    cur_data_type = np.dtype(cur_data_type)

    n_training_points = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5, lambda d: len(d[is_training_point_f(d)]), 0)
    # print(f'[get_features_sets] non-empty points: {n_training_points}')
    cur_h5_dset = cur_h5.create_dataset('c', (n_training_points, ),
                                        dtype=cur_data_type,
                                        chunks=(n_training_points / 10, ))
    t1 = time()
    if profiling:
        print(f'[get_features_sets] cur_create_time: {t1-t0}')

    # Copy porosity data to cur structure
    # Only copy points which will be used for training, i.e., not empty.
    # Deep copy is required since porosity_data_h5 has all points
    # (including empty points) which won't be used for training, and
    # just filtering these out would return a ndarray in-memory structure.
    # This ndarray can be too large to fit in memory.
    prev_end = 0
    hypercube_shape = porosity_data_h5.shape
    for chunk_slice in porosity_data_h5.iter_chunks():
        # Get current chunk
        chunk_np = porosity_data_h5[chunk_slice]

        # Append these porosity values to the current dataset
        training_points = chunk_np[is_training_point_f(chunk_np)]
        if features_only:
            cur_h5_dset['x', 'y', 'z', 'phi',
                        prev_end:(prev_end +
                                  len(training_points))] = training_points[[
                                      'x', 'y', 'z', 'phi'
                                  ]]
        else:
            cur_h5_dset['x', 'y', 'z', 'phi', 'well_id',
                        prev_end:(prev_end +
                                  len(training_points))] = training_points[[
                                      'x', 'y', 'z', 'phi', 'well_id'
                                  ]]
        prev_end = prev_end + len(training_points)

    t2 = time()
    if profiling:
        print(f'[get_features_sets] cur_copy_porosity_time: {t2-t1}')

    return cur_h5, cur_h5_dset


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(
        porosity_data_h5,
        features_dict_h5,
        all_features,
        displacement_cube_shape,
        # num_threads,
        it_str,
        exp_n_features,
        f_width=0):

    t0 = time()

    # Points used for training: real, expanded and propagated
    is_training_point_f = lambda d: (
        (d['real'] == common.RealValues.real) |
        (d['real'] == common.RealValues.canal_expanded) |
        (d['real'] == common.RealValues.expanded) |
        (d['real'] == common.RealValues.propagated))

    cur_h5, cur_h5_dset = create_tmp_dset(porosity_data_h5,
                                          is_training_point_f, exp_n_features)

    # Create sequence object
    cur_h5_seq = hdf5_util.HDFMultiColSequence(cur_h5_dset, ['x', 'y', 'z'])

    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']

    # List of features sets and their error metric
    results = []

    hypercube_shape = porosity_data_h5.shape

    # Find a feature set with exp_n_features features
    for it in range(exp_n_features):

        t3 = time()
        # Reset best feature and its error
        best_error = 10000
        best_feature = ()

        # Setup the new column to be tested
        cur_h5_seq.add_new_col()

        # Test each available feature
        ii = 0
        for cur_feature in all_features:
            t4 = time()
            if cur_feature in cur_f_set:
                continue

            # Early termination for debugging
            if f_width != 0 and ii == f_width:
                break
            ii = ii + 1

            # Insert a temporary feature
            insert_filtered_feature(cur_h5_dset, cur_h5_seq, features_dict_h5,
                                    cur_feature, hypercube_shape,
                                    displacement_cube_shape)
            t5 = time()
            print(f'[get_features_sets][{cur_feature}] '\
                  f'insert_feature_time: {t5-t4}')

            # Test the model with cur_feature
            # rmse, mae = eval_bootstrap(cur_h5_seq, list(range(1,11)))
            rmse, mae = eval_bootstrap(cur_h5_seq, list(range(10)))
            t6 = time()
            print(f'[get_features_sets][{cur_feature}] '\
                  f'train_time: {t6-t5}')
            print(f'[get_features_sets][{cur_feature}] '\
                  f'error: {rmse}')

            results.append((cur_f_set + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            print(f'[get_features_sets][it{it}] Tested features '\
                  f'{cur_f_set+ [cur_feature]} with error {rmse}')

        # Remove the best feature from the features list
        all_features.remove(best_feature)

        t7 = time()
        print(f'[get_features_sets][it{it}] '\
              f'full_it_time: {t7-t3}')

        # Update the last column of the sequence object to the best feature
        insert_filtered_feature(cur_h5_dset, cur_h5_seq, features_dict_h5,
                                best_feature, hypercube_shape,
                                displacement_cube_shape)
        cur_f_set.append(best_feature)

        t8 = time()
        print(f'[get_features_sets][{cur_feature}] '\
              f'commit_feature_time: {t8-t7}')

    t9 = time()
    print(f'[get_features_sets] full_time: {t9-t0}')

    cur_h5.close()

    return get_best_features_set(results)
