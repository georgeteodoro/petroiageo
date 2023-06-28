import numpy as np
import random
import sys
from time import time
import lightgbm as lgb
from math import prod

import common
import petro5_hdf5
import hdf5_util
import profiling

# Parameters
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


def perf_predition(
    best_features_set,
    porosity_data_h5,
    features_dict_h5,
    window_sizes,
    displacement_cube_shape,
    it,
    config,
):
    t0 = time()

    # Remove coordinates from features set
    best_features_set.remove("x")
    best_features_set.remove("y")
    best_features_set.remove("z")

    # Points used for training: real and propagated
    is_training_point_f = lambda d: (
        (d["real"] == common.RealValues.real)
        | (d["real"] == common.RealValues.propagated)
    )

    # Creates a temporary h5 structure to perform the training
    cur_h5, cur_h5_dset = petro5_hdf5.create_tmp_dset(
        porosity_data_h5, is_training_point_f, len(best_features_set), True
    )
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)

    hypercube_shape = porosity_data_h5.shape

    t1 = time()
    profiling.prof_predict_create_time(it, t1 - t0, config)

    # Add each feature to the DataFrame
    for feature in best_features_set:
        cur_h5_train_list.add_new_col()
        petro5_hdf5.insert_filtered_feature(
            cur_h5_dset,
            cur_h5_train_list,
            features_dict_h5,
            feature,
            window_sizes,
            hypercube_shape,
            displacement_cube_shape,
        )

    t2 = time()
    profiling.prof_predict_insert_time(
        it, len(best_features_set), t2 - t1, config
    )

    regressor = None
    for c in range(cur_h5_train_list.n_chunks):
        # Generate a training dataset for all data on chunk c without
        # data from well w
        X_train_np, y_train_np = cur_h5_train_list.get_all_well_data(c)
        lgb_train_dataset = lgb.Dataset(X_train_np, y_train_np)

        # Perform training
        regressor = lgb.train(
            params,
            lgb_train_dataset,
            init_model=regressor,
            num_boost_round=100,
            keep_training_booster=True,
        )

    cur_h5.close()

    t3 = time()
    profiling.prof_predict_train_times(it, t3 - t2, config)

    total_to_propagate = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5,
        lambda d: len(
            d[
                (d["real"] == common.RealValues.canal_expanded)
                | (d["real"] == common.RealValues.expanded)
            ]
        ),
        0,
    )

    # Perform prediction of expanded points
    p_sum = 0
    chunk_n = -1
    for cur_slice in porosity_data_h5.iter_chunks():
        t4 = time()
        chunk_n += 1

        cur_chunk_np = porosity_data_h5[cur_slice]

        # Define a function to filter only the expanded points
        is_to_pred_point_f = lambda d: (
            (d["real"] == common.RealValues.canal_expanded)
            | (d["real"] == common.RealValues.expanded)
        )

        # Check if there is any point on the current chunk to be updated
        # TODO:
        # Extract check for overlapping rectangles in expand.gen_expanded_points,
        # parameterize by (it, window_size, slice) and use it here
        if not is_to_pred_point_f(cur_chunk_np).any():
            continue

        # Filter points to predict
        expanded_points_np = cur_chunk_np[is_to_pred_point_f(cur_chunk_np)]

        to_propagate_count = len(expanded_points_np)
        p_sum = p_sum + to_propagate_count

        # Get coordinates of points to predict
        coords_3d_np = expanded_points_np[["x", "y", "z"]]

        # Create new ndarray for keeping all features values
        # of the current chunk
        predict_features_type = [
            (f"f{f}", np.float64) for f in range(len(best_features_set))
        ]
        to_predict_np = np.empty(
            to_propagate_count, dtype=predict_features_type
        )

        # Function to filter features with a given coords list
        def _gen_list_features(feature_dset, coords_3d_np):
            for coord in coords_3d_np:
                yield feature_dset[tuple(coord)]

        # Fill features values
        i = 0
        for feature in best_features_set:
            # Apply the displacement
            cur_coords_3d_np = coords_3d_np.copy()
            for coord_s, d_id in [("x", 0), ("y", 1), ("z", 2)]:
                cur_coords_3d_np[coord_s] = (
                    cur_coords_3d_np[coord_s]
                    + feature[d_id + 1]
                    + ((displacement_cube_shape[d_id] - 1) / 2)
                )

            # Filter features' values for current chunk coords
            feature_values = _gen_list_features(
                features_dict_h5[feature[0]], cur_coords_3d_np.flat
            )

            to_predict_np[f"f{i}"] = np.fromiter(feature_values, np.float64)

            i = i + 1

        # Convert to_predict_np from a ndarray to a regular 2d array
        to_predict_np = np.array(to_predict_np.tolist())

        t5 = time()
        profiling.prof_predict_pred_insert_time(it, chunk_n, t5 - t4, config)

        # Perform prediction of expanded points
        new_phi_np = regressor.predict(to_predict_np)

        t6 = time()
        profiling.prof_predict_pred_run_time(it, chunk_n, t6 - t5, config)

        # Update porosity values on cur_chunk_np. This serves 2 purposes:
        # (i) converts the new_phi_np from a list to a 3d array (cur_chunk_np)
        # on the proper coordinates and, (ii) allows batched write of data
        # onto porosity_data_h5
        updated_coords = np.where(is_to_pred_point_f(cur_chunk_np))
        cur_chunk_np["phi"][updated_coords] = new_phi_np
        cur_chunk_np["real"][updated_coords] = common.RealValues.propagated

        # Only update 'phi' and 'real' values of expanded points
        porosity_data_h5[
            "phi", cur_slice[0], cur_slice[1], cur_slice[2]
        ] = cur_chunk_np["phi"]
        porosity_data_h5[
            "real", cur_slice[0], cur_slice[1], cur_slice[2]
        ] = cur_chunk_np["real"]

        t7 = time()
        profiling.prof_predict_pred_update_time(it, chunk_n, t7 - t6, config)
        profiling.prof_predict_pred_time(it, chunk_n, t7 - t4, config)

    t8 = time()
    profiling.prof_predict_pred_times(it, t8 - t3, config)
