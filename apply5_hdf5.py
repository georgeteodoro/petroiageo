import numpy as np
import random
import sys
from time import time
import lightgbm as lgb
from math import prod

# from numba import jit

import common
import petro4_hdf5
import hdf5_util

# Parameters
LABEL_COLUMN_NAME = 'phi'
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
    "random_state": 0
}

SEISMIC_MAX_X = 433
SEISMIC_MAX_Y = 645
SEISMIC_MAX_Z = 250

# def eval_model(orig_df, main_df, features):

#     # Get points marked for prediction
#     X_to_predict = main_df[(main_df['real'] == 2) | (main_df['real'] == 3)]
#     X_to_predict = X_to_predict[[common.f2str(f) for f in features]]

#     # Only train on points with phi value
#     X_with_phi = main_df[(main_df['real'] == 0) | (main_df['real'] == 1)]
#     X_with_phi = X_with_phi[[common.f2str(f) for f in features]]

#     # Results (phi) only for predicted or original points
#     y_df = main_df[(main_df['real'] == 0) |
#                    (main_df['real'] == 1)][LABEL_COLUMN_NAME]

#     # Train
#     lgb_train = lgb.Dataset(X_with_phi.values, y_df.values)
#     regressor = lgb.train(
#         params,
#         lgb_train,
#         num_boost_round=100,
#     )

#     # Predict expanded points
#     pred = regressor.predict(X_to_predict.values)

#     # Get the two disjoint set of points, real + previously expanded
#     # and expanded on this iteration
#     remaining_df = orig_df[(orig_df['real'] != 2) & (orig_df['real'] != 3)]
#     predicted_df = orig_df[(orig_df['real'] == 2) | (orig_df['real'] == 3)]

#     # SettingWithCopyWarning is false positive on the two .loc lines below
#     pd.options.mode.chained_assignment = None

#     # Update real value from 3 (to predict) and 2 (to expand) to 1 (propagated)
#     predicted_df.loc[:, 'real'] = 1

#     # Assign predicted values
#     predicted_df.loc[:, 'phi'] = pred

#     # Re-enable SettingWithCopyWarning
#     pd.options.mode.chained_assignment = 'warn'

#     return pd.concat([remaining_df, predicted_df])


def get_feature_col(features_dict_h5, feature, coords_3d_np, hypercube_shape,
                    displacement_cube_shape):
    coord_planar_np = coords_3d_np.copy()

    # Convert 3d coords to planar coords
    for (coord_s, d_id) in [('x', 0), ('y', 1), ('z', 2)]:
        coord_planar_np[coord_s] = coord_planar_np[coord_s] + feature[
            d_id + 1] + ((displacement_cube_shape[d_id] - 1) / 2)
    displaced_hypercube_shape = np.array(hypercube_shape) + (
        np.array(displacement_cube_shape) - 1)
    coord_planar_np = coord_planar_np[
        'z'].flat + coord_planar_np['y'].flat * displaced_hypercube_shape[
            2] + coord_planar_np['x'].flat * displaced_hypercube_shape[
                2] * displaced_hypercube_shape[1]

    coord_planar_np.sort()

    return features_dict_h5[feature[0]][coord_planar_np]


def perf_predition(best_features_set, porosity_data_h5, features_dict_h5,
                   displacement_cube_shape):
    profiling = 0

    t1 = time()

    # Remove coordinates from features set
    best_features_set.remove('x')
    best_features_set.remove('y')
    best_features_set.remove('z')

    # # Create a DataFrame for the features to be used for prediction
    # cur_features_df = main_df.copy(deep=False)

    # Points used for training: real and propagated
    is_training_point_f = lambda d: (
        (d['real'] == common.RealValues.real) |
        (d['real'] == common.RealValues.propagated))

    # Creates a temporary h5 structure to perform the training
    cur_h5, cur_h5_dset = petro4_hdf5.create_tmp_dset(porosity_data_h5,
                                                      is_training_point_f,
                                                      len(best_features_set),
                                                      True)
    cur_h5_seq = hdf5_util.HDFMultiColSequence(cur_h5_dset, [])

    hypercube_shape = porosity_data_h5.shape

    # Add each feature to the DataFrame
    for feature in best_features_set:
        cur_h5_seq.add_new_col()
        petro4_hdf5.insert_filtered_feature(cur_h5_dset, cur_h5_seq,
                                            features_dict_h5, feature,
                                            hypercube_shape,
                                            displacement_cube_shape)
    cur_h5_seq.update_len()

    t2 = time()
    if profiling > 0:
        print(f'[apply5_hdf5] prep-tmp-data: {t2-t1}')

    # Setup the training dataset
    X_train_seq = cur_h5_seq
    y_train_np = cur_h5_seq.get_y_np()
    lgb_train_dataset = lgb.Dataset(X_train_seq, y_train_np)

    # Perform training
    regressor = lgb.train(params, lgb_train_dataset, num_boost_round=100)

    cur_h5.close()
    t3 = time()
    if profiling > 0:
        print(f'[apply5_hdf5] trained: {t3-t2}')

    total_to_propagate = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5,
        lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                        (d['real'] == common.RealValues.expanded)]), 0)

    # Perform prediction of expanded points
    # p_sum = 0
    for cur_slice in porosity_data_h5.iter_chunks():
        t31 = time()

        cur_chunk = porosity_data_h5[cur_slice]

        # Check if there is any point on the current chunk to be updated
        to_propagate = len(
            cur_chunk[(cur_chunk['real'] == common.RealValues.canal_expanded) |
                      (cur_chunk['real'] == common.RealValues.expanded)])
        if to_propagate == 0:
            continue
        # p_sum = p_sum + to_propagate
        # print(f'to_propagate: {to_propagate}')
        # print(f'propagating {p_sum}/{total_to_propagate} points')

        # Create new ndarray for keeping all features values
        # of the current chunk
        slice_len = prod([s.stop - s.start for s in cur_slice])
        predict_features_type = [(f'f{f}', np.float64)
                                 for f in range(len(best_features_set))]
        to_predict_np = np.empty(slice_len, dtype=predict_features_type)

        # Fill features values
        i = 0
        for feature in best_features_set:
            to_predict_np[f'f{i}'] = get_feature_col(
                features_dict_h5, feature, cur_chunk[['x', 'y', 'z']],
                hypercube_shape, displacement_cube_shape)

            i = i + 1
        t32 = time()
        if profiling > 1:
            print(f'[apply5_hdf5] fill-features {t32-t31}')

        # Convert to_predict_np from a ndarray to a regular 2d array
        to_predict_np = np.array(to_predict_np.tolist())

        # Perform prediction on all points of the chunk (even ones that don't
        # need prediction)
        new_phi_np = regressor.predict(to_predict_np)
        new_phi_np = new_phi_np.reshape(cur_chunk.shape)

        t33 = time()
        if profiling > 1:
            print(f'[apply5_hdf5] predicted {t33-t32}')

        # Define a function to filter only the expanded points
        is_to_pred_point_f = lambda d: (
            (d['real'] == common.RealValues.canal_expanded) |
            (d['real'] == common.RealValues.expanded))

        # Only update 'phi' and 'real' values of expanded points
        porosity_data_h5['phi', cur_slice[0], cur_slice[1],
                         cur_slice[2]] = np.where(
                             is_to_pred_point_f(cur_chunk), new_phi_np,
                             cur_chunk['phi'])

        t34 = time()
        if profiling > 1:
            print(f'[apply5_hdf5] update-phi {t34-t33}')

        porosity_data_h5['real', cur_slice[0], cur_slice[1],
                         cur_slice[2]] = np.where(
                             is_to_pred_point_f(cur_chunk),
                             common.RealValues.propagated, cur_chunk['real'])

        t35 = time()
        if profiling > 1:
            print(f'[apply5_hdf5] update-real {t35-t34}')
            print(f'[apply5_hdf5] done-in {t35-t31}')

    # # Train model and predict porosity for new expanded points
    # ret = eval_model(main_df, cur_features_df, best_features_set)
    t4 = time()
    if profiling > 0:
        print(f'[apply5_hdf5] updated-values: {t4-t3}')
