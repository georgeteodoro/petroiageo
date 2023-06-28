import numpy as np
from time import time
from typing import Tuple, Dict
import h5py
from math import prod
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

import hdf5_util
import common
from config_parser import Config

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

# This version only works for the first iteration.
# The tmp dataset is a list based on the sparse data of the
# porosity dataset. It can either be incremental or need to be
# reassembled each iteration. If incremental, indices will be
# mismatched between features and tmp list. If reassembling,
# it will be inefficient.


def get_best_features_set(
    features_sets: list[tuple[list, float]]
) -> Tuple[list, float]:
    # Sort by second column (id 1)
    features_sets.sort(key=lambda tup: tup[1])
    best_features_set = features_sets[0][0]
    best_error = features_sets[0][1]

    return best_features_set, best_error


def eval_bootstrap(
    cur_h5_train_list: hdf5_util.HDFMultiColList,
    cur_h5_test_list: hdf5_util.HDFMultiColList,
    wells_id: list[int],
    num_threads=1,
) -> Tuple[float, float]:
    """
    Train a regressor on cur_h5_train_list using data of wells in wells_id.
    If cur_h5_test_list == None, then test errors will be based on the
    validation data for each Leave-one-well-out iteration.
    Return the mean rmse and mean mae errors
    """
    params["num_threads"] = num_threads

    profiling = False

    rmse_list, mae_list = _leave_one_well_out_training(
        cur_h5_train_list, cur_h5_test_list, wells_id, profiling
    )

    return np.mean(rmse_list), np.mean(mae_list)


def _leave_one_well_out_training(
    cur_h5_train_list: hdf5_util.HDFMultiColList,
    cur_h5_test_list: hdf5_util.HDFMultiColList,
    wells_id: int,
    profiling: bool,
) -> Tuple[list[float], list[float]]:
    # Test data is the same for all wells if test_only_wells are
    # active, so it's only setup once
    if cur_h5_test_list is not None:
        X_test, y_test = cur_h5_test_list.get_data_not_in_well()

    t0 = time()
    rmse_list: list[float] = []
    mae_list: list[float] = []

    # Leave-One-Well-Out
    for curr_well_id in wells_id:
        X_val, y_val, regressor = _incremental_learning(
            cur_h5_train_list, profiling, curr_well_id
        )
        t3 = time()

        # Calculate error metrics
        if cur_h5_test_list == None:
            X_test = X_val
            y_test = y_val

        pred = regressor.predict(X_test)
        rmse = np.sqrt(np.mean((pred - y_test) ** 2))
        mae = mean_absolute_error(pred, y_test)
        rmse_list.append(rmse)
        mae_list.append(mae)

        if profiling:
            t4 = time()
            print(
                f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Evaluating in {t4-t3}"
            )
            print(
                f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Total time {t4-t0}"
            )

    return rmse_list, mae_list


def _incremental_learning(
    cur_h5_train_list: hdf5_util.HDFMultiColList,
    profiling: bool,
    curr_well_id: int,
) -> Tuple[np.ndarray, np.ndarray, lgb.Booster]:
    """
    See discussion for incremental learning:
    https://stackoverflow.com/questions/73664093/lightgbm-train-vs-update-vs-refit
    """
    # Extract the validation data
    # Since the same validation data is supposed to be used for
    # all incremental trainings and is small enough to fit in
    # memory
    X_val, y_val = cur_h5_train_list.get_data_from_well(curr_well_id)

    # Incremental training on all cur_h5_train_list chunks
    regressor = None
    setup_time = 0
    training_time = 0
    for chunk_idx in range(cur_h5_train_list.n_chunks):
        t1 = time()

        X_train, y_train = cur_h5_train_list.get_data_not_in_well(
            chunk_idx, curr_well_id
        )
        lgb_train_dataset = lgb.Dataset(X_train, y_train)

        lgb_eval_dataset = lgb.Dataset(
            X_val, y_val, reference=lgb_train_dataset
        )

        t2 = time()

        regressor = lgb.train(
            params,
            lgb_train_dataset,
            init_model=regressor,
            num_boost_round=100,
            valid_sets=lgb_eval_dataset,
            keep_training_booster=True,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        t3 = time()

        setup_time += t2 - t1
        training_time += t3 - t2

        if profiling:
            print(
                f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Setup in "
                f"{t2 - t1}"
            )
            print(
                f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Training in "
                f"{t3 - t2}"
            )

    if profiling:
        print(
            f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Final setup in "
            f"{setup_time}"
        )
        print(
            f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Final training in "
            f"{training_time}"
        )

    return X_val, y_val, regressor


def insert_filtered_feature(
    cur_h5_dset: h5py.Dataset,
    cur_h5_seq: hdf5_util.HDFMultiColList,
    features_dict_h5: Dict[str, h5py.Dataset],
    cur_feature: tuple,
    hypercube_shape,
    displacement_cube_shape: tuple,
):
    profile_time = False

    t0 = time()
    chunk_start = 0
    for list_chunk_slice in cur_h5_dset.iter_chunks():
        t1 = time()
        chunk_np = cur_h5_dset[list_chunk_slice]
        # print(f'[insert_filtered_feature] cur slice: {list_chunk_slice}')
        # print(f'[insert_filtered_feature] chunk size: {len(chunk_np)}')

        # Get the coordinates list
        coord_3d_np = chunk_np[["x", "y", "z"]]

        t2 = time()
        if profile_time:
            print(f"[insert_filtered_feature] get_slice_time: {t2-t1}")

        # Apply the displacement
        for coord_s, d_id in [("x", 0), ("y", 1), ("z", 2)]:
            coord_3d_np[coord_s] = (
                coord_3d_np[coord_s]
                + cur_feature[d_id + 1]
                + ((displacement_cube_shape[d_id] - 1) / 2)
            )

        t3 = time()
        if profile_time:
            print(f"[insert_filtered_feature] appply_disp_time: {t3-t2}")

        # We use a generator in order to avoid copying the displaced feature
        # data into a ndarray variable, to later copy it to the cur_h5_seq
        # object.
        def _feature_generator(feature_dset, coord_3d_np, chunk_start):
            def _gen_list_features(feature_dset, coord_3d_np):
                for coord in coord_3d_np:
                    yield feature_dset[tuple(coord)]

            # Append slice of current chunk to features list
            yield slice(
                chunk_start, chunk_start + len(coord_3d_np)
            ), _gen_list_features(feature_dset, coord_3d_np)

        feature_gen = _feature_generator(
            features_dict_h5[cur_feature[0]], coord_3d_np, chunk_start
        )

        chunk_start += list_chunk_slice[0].stop - list_chunk_slice[0].start - 1

        t4 = time()
        if profile_time:
            print(f"[insert_filtered_feature] generator_time: {t4-t3}")

        # Insert the generator displaced feature data into cur_h5_seq
        cur_h5_seq.update_last_col_chunk(feature_gen)

        t5 = time()
        if profile_time:
            print(
                f"[insert_filtered_feature] insert_disp_feature_time: {t5-t4}"
            )

    t6 = time()
    if profile_time:
        print(f"[insert_filtered_feature] full_time: {t6-t0}")


def create_tmp_dset(
    porosity_data_h5: h5py.Dataset,
    is_training_point_f,
    n_features,
    suf_str="",
    list_chunk_size=1000,
    features_only=False,
    test_only_wells=[],
) -> Tuple[h5py.File, h5py.Dataset, h5py.File, h5py.Dataset]:
    profiling = False

    filename = f"cur{suf_str}.h5"
    filename_test = f"cur{suf_str}-test.h5"

    t0 = time()
    # If the cur file exists, it should be deleted
    # A new tmp file is created by iteration
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(filename_test):
        os.remove(filename_test)

    # Creates a temporary h5 structure to maintain the porosity
    # and features data
    cur_h5 = h5py.File(f"{filename}", "w")

    if features_only:
        cur_data_type = [
            ("x", np.int64),
            ("y", np.int64),
            ("z", np.int64),
            ("phi", np.float64),
        ]
    else:
        cur_data_type = [
            ("x", np.int64),
            ("y", np.int64),
            ("z", np.int64),
            ("phi", np.float64),
            ("well_id", np.int64),
        ]

    cur_data_type = cur_data_type + [
        (f"f{f}", np.float64) for f in range(n_features)
    ]
    cur_data_type = np.dtype(cur_data_type)

    def _is_well_in_list(d, l):
        ret = np.full((d.shape), False, dtype=bool)
        for x in l:
            ret += d["well_id"] == x
        return ret

    def _is_well_not_in_list(d, l):
        ret = np.full((d.shape), True, dtype=bool)
        for x in l:
            ret *= d["well_id"] != x
        return ret

    # Select whether training points include all points or there are test
    # points as well
    is_training_point_f2 = is_training_point_f
    if len(test_only_wells) > 0:
        is_training_point_f2 = lambda d: is_training_point_f(
            d
        ) & _is_well_not_in_list(d, test_only_wells)

        n_test_points = hdf5_util.fold_h5_all_clusters(
            porosity_data_h5,
            lambda d: _is_well_in_list(d, test_only_wells).sum(),
            0,
        )

    n_training_points = hdf5_util.fold_h5_all_clusters(
        porosity_data_h5, lambda d: len(d[is_training_point_f2(d)]), 0
    )

    print("============== NEED TO AUTOMATE TMP_LIST CHUNK_SIZE")
    # cur_chunksize = (n_training_points / 10, )
    cur_chunksize = (n_training_points,)

    cur_h5_dset = cur_h5.create_dataset(
        "c", (n_training_points,), dtype=cur_data_type, chunks=cur_chunksize
    )
    test_h5 = None
    test_h5_dset = None
    if len(test_only_wells) > 0:
        test_h5 = h5py.File(f"{filename_test}", "w")
        test_h5_dset = test_h5.create_dataset(
            "c", (n_test_points,), dtype=cur_data_type, chunks=(n_test_points,)
        )

    t1 = time()
    if profiling:
        print(f"[get_features_sets] cur_create_time: {t1-t0}")

    # Copy porosity data to cur structure
    # Only copy points which will be used for training, i.e., not empty.
    # Deep copy is required since porosity_data_h5 has all points
    # (including empty points) which won't be used for training, and
    # just filtering these out would return a ndarray in-memory structure.
    # This ndarray can be too large to fit in memory.
    prev_end = 0
    prev_end_test = 0
    hypercube_shape = porosity_data_h5.shape
    for chunk_slice in porosity_data_h5.iter_chunks():
        # Get current chunk
        chunk_np = porosity_data_h5[chunk_slice]

        # Extract training points from chunk
        training_points = chunk_np[is_training_point_f(chunk_np)]

        if len(test_only_wells) > 0:
            # Filer out training points, removing the test data
            training_points = training_points[
                _is_well_not_in_list(training_points, test_only_wells)
            ]

            # Add test data to its unique list
            test_points = training_points[
                _is_well_in_list(training_points, test_only_wells)
            ]
            test_h5_dset[
                "x",
                "y",
                "z",
                "phi",
                prev_end_test : (prev_end_test + len(test_points)),
            ] = test_points[["x", "y", "z", "phi"]]
            prev_end_test += len(test_points)

        # Append these porosity values to the current dataset
        if features_only:
            cur_h5_dset[
                "x",
                "y",
                "z",
                "phi",
                prev_end : (prev_end + len(training_points)),
            ] = training_points[["x", "y", "z", "phi"]]
        else:
            cur_h5_dset[
                "x",
                "y",
                "z",
                "phi",
                "well_id",
                prev_end : (prev_end + len(training_points)),
            ] = training_points[["x", "y", "z", "phi", "well_id"]]
        prev_end += len(training_points)

    t2 = time()
    if profiling:
        print(f"[get_features_sets] cur_copy_porosity_time: {t2-t1}")
        print(f"[get_features_sets] final_lenght: {cur_h5_dset.shape}")

    return cur_h5, cur_h5_dset, test_h5, test_h5_dset


# exp_n_features: number of features to be selected
# max_tested_features: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(
    porosity_data_h5: h5py.Dataset,
    features_dict_h5: Dict[str, h5py.Dataset],
    all_features: list,
    displacement_cube_shape: tuple,
    it: int,
    exp_n_features: int,
    max_tested_features: int,
    config: Config,
):
    t0 = time()

    # Points used for training: real and propagated
    is_training_point_f = lambda d: (
        (d["real"] == common.RealValues.real)
        | (d["real"] == common.RealValues.canal_expanded)
        | (d["real"] == common.RealValues.propagated)
    )

    test_only_wells = config.alg["test_only_wells"]
    wells_coords = config.wells["coords"]
    training_wells = list(range(len(wells_coords)))
    training_wells = [x for x in training_wells if x not in test_only_wells]

    cur_h5, cur_h5_dset, test_h5, test_h5_dset = create_tmp_dset(
        porosity_data_h5,
        is_training_point_f,
        exp_n_features,
        test_only_wells=test_only_wells,
    )

    # Create training temporary object
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)
    cur_h5_test_list = None
    if len(test_only_wells) > 0:
        cur_h5_test_list = hdf5_util.HDFMultiColList(test_h5_dset)

    # Current features set with the best error
    cur_f_set = ["x", "y", "z"]

    # List of features sets and their error metric
    results = []

    hypercube_shape = porosity_data_h5.shape

    # Find a feature set with exp_n_features features
    for it in range(exp_n_features):
        t3 = time()
        # Reset best feature and its error
        best_error = float("inf")
        best_feature = None

        # Setup the new column to be tested
        cur_h5_train_list.add_new_col()
        if len(test_only_wells) > 0:
            cur_h5_test_list.add_new_col()

        # Test each available feature
        ii = 0
        for cur_feature in all_features:
            t4 = time()
            if cur_feature in cur_f_set:
                continue

            # Early termination for debugging
            if max_tested_features != 0 and ii == max_tested_features:
                break
            ii = ii + 1

            # Insert a temporary feature
            insert_filtered_feature(
                cur_h5_dset,
                cur_h5_train_list,
                features_dict_h5,
                cur_feature,
                hypercube_shape,
                displacement_cube_shape,
            )

            # Also inserts the feature on the test dataset, if necessary
            if len(test_only_wells) > 0:
                insert_filtered_feature(
                    test_h5_dset,
                    cur_h5_test_list,
                    features_dict_h5,
                    cur_feature,
                    hypercube_shape,
                    displacement_cube_shape,
                )

            t5 = time()
            print(
                f"[get_features_sets][{cur_feature}] "
                f"insert_feature_time: {t5-t4}"
            )

            # Test the model with cur_feature
            rmse, mae = eval_bootstrap(
                cur_h5_train_list, cur_h5_test_list, training_wells
            )
            t6 = time()
            print(f"[get_features_sets][{cur_feature}] " f"train_time: {t6-t5}")
            print(f"[get_features_sets][{cur_feature}] " f"error: {rmse}")

            results.append((cur_f_set + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            print(
                f"[get_features_sets][it{it}] Tested features "
                f"{cur_f_set+ [cur_feature]} with error {rmse}"
            )

        # Remove the best feature from the features list
        all_features.remove(best_feature)

        t7 = time()
        print(f"[get_features_sets][it{it}] " f"full_it_time: {t7-t3}")

        # Update the last column of the sequence object to the best feature
        insert_filtered_feature(
            cur_h5_dset,
            cur_h5_train_list,
            features_dict_h5,
            best_feature,
            hypercube_shape,
            displacement_cube_shape,
        )

        # Also inserts the feature on the test dataset, if necessary
        if len(test_only_wells) > 0:
            insert_filtered_feature(
                test_h5_dset,
                cur_h5_test_list,
                features_dict_h5,
                best_feature,
                hypercube_shape,
                displacement_cube_shape,
            )

        cur_f_set.append(best_feature)

        t8 = time()
        print(
            f"[get_features_sets][{best_feature}] "
            f"commit_feature_time: {t8-t7}"
        )

    t9 = time()
    print(f"[get_features_sets] full_time: {t9-t0}")

    cur_h5.close()
    test_h5.close()

    return get_best_features_set(results)
