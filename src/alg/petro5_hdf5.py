import numpy as np
from time import time
from typing import Tuple, Dict
import h5py
import os

from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

import hdf5_util
from config_parser import Config
from data_filter import WellsDataFilter, DataFilter
from data_filter import FeatSelectionTrainDataFilter

RANDOM_STATE = 1
MAX_HDF5_CHUNK_SIZE = 4_294_967_296  #2 ** 32
TMP_DSET_NAME = 'c'

params = {
    'max_bin': 128,
    'max_depth': 10,
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 20,
    'verbose': -1,
    'min_data': 10,
    'boost_from_average': True,
    'bagging_freq': 1,
    'random_state': RANDOM_STATE,
    # 'tree_learner': 'data',
}


def get_best_features_set(
        features_sets: list[tuple[list, float]]) -> Tuple[list, float]:
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
    params['num_threads'] = num_threads

    profiling = False

    rmse_list, mae_list = _leave_one_well_out_training(cur_h5_train_list,
                                                       cur_h5_test_list,
                                                       wells_id, profiling)

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
            cur_h5_train_list, profiling, curr_well_id)
        t3 = time()

        # Calculate error metrics
        if cur_h5_test_list == None:
            X_test = X_val
            y_test = y_val

        pred = regressor.predict(X_test)
        rmse = np.sqrt(np.mean((pred - y_test)**2))
        mae = mean_absolute_error(pred, y_test)
        rmse_list.append(rmse)
        mae_list.append(mae)

        if profiling:
            t4 = time()
            print(f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] "\
                  f"Evaluating in {t4-t3}")
            print(f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] "\
                  f"Total time {t4-t0}")

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
            chunk_idx, curr_well_id)
        lgb_train_dataset = lgb.Dataset(X_train, y_train)

        lgb_eval_dataset = lgb.Dataset(X_val,
                                       y_val,
                                       reference=lgb_train_dataset)

        if len(X_val) == 0 | len(y_val) == 0:
            raise Exception(f"[petro5_hdf5] Bad data: well_id {curr_well_id} "\
                             "have no points on porosity dataset.")

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
            print(f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Setup in "
                  f"{t2 - t1}")
            print(
                f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Training in "
                f"{t3 - t2}")

    if profiling:
        print(f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Final setup in "
              f"{setup_time}")
        print(
            f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Final training in "
            f"{training_time}")

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
        coord_3d_np = chunk_np[['x', 'y', 'z']]

        t2 = time()
        if profile_time:
            print(f"[insert_filtered_feature] get_slice_time: {t2-t1}")

        # Apply the displacement
        for coord_s, d_id in [('x', 0), ('y', 1), ('z', 2)]:
            coord_3d_np[coord_s] = (coord_3d_np[coord_s] +
                                    cur_feature[d_id + 1] +
                                    ((displacement_cube_shape[d_id] - 1) / 2))

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
            yield slice(chunk_start,
                        chunk_start + len(coord_3d_np)), _gen_list_features(
                            feature_dset, coord_3d_np)

        feature_gen = _feature_generator(features_dict_h5[cur_feature[0]],
                                         coord_3d_np, chunk_start)

        chunk_start += list_chunk_slice[0].stop - list_chunk_slice[0].start - 1

        t4 = time()
        if profile_time:
            print(f"[insert_filtered_feature] generator_time: {t4-t3}")

        # Insert the generator displaced feature data into cur_h5_seq
        cur_h5_seq.update_last_col_chunk(feature_gen)

        t5 = time()
        if profile_time:
            print(
                f"[insert_filtered_feature] insert_disp_feature_time: {t5-t4}")

    t6 = time()
    if profile_time:
        print(f"[insert_filtered_feature] full_time: {t6-t0}")


def _add_sampling_window_to_filters(sampling_window: int,
                                    train_data_filter: DataFilter,
                                    test_data_filter: DataFilter,
                                    it) -> Tuple[DataFilter, DataFilter]:
    """
    Updates the train and test filters based on sampling_window
    """
    min_ring = max(0, it - sampling_window)

    train_data_filter.add_min_ring_filter(min_ring)
    test_data_filter.add_min_ring_filter(min_ring)

    print(f"added min ring: {min_ring}")
    return train_data_filter, test_data_filter


def _limit_training_points(sampling_max_points: int,
                           curr_n_train_points: int) -> int:
    """
    Defines the total of training points required.
    If sampling_max_points > 0 and it is lower then curr_n_train_points,
    then it will be forced. A negative sampling_max_points means no sampling.
    """
    if sampling_max_points > 0:
        curr_n_train_points = min(curr_n_train_points, sampling_max_points)
    return curr_n_train_points


def _prepare_h5(suf_str: str,
                test_only_wells: list,
                features_only: bool,
                n_features: int,
                train_data_filter: DataFilter,
                test_data_filter: DataFilter,
                porosity_data_h5: h5py.Dataset,
                it: int,
                config: Config,
                should_sample_max_points: bool = True):
    """
    Generate the h5 File and dataset objects.
    Also setup the filter functions to return only training and test points 
    and returns the total number of sampled training points. If no sampling is
    done, returns all training and test points.
    If should_sample_max_points, then considers the max_points param of the 
    sampling param in config. If not, then no sampling of max points
    is used
    """

    train_data_filter, test_data_filter = _config_filters(
        test_only_wells, train_data_filter, test_data_filter, it,
        config.alg['sampling']['its_window_size'])

    print(f"TRAIN_FILTER: {train_data_filter}")
    print(f"TEST_FILTER: {test_data_filter}")

    #We may not have the sampling window and still have
    #sampling max points defined
    if should_sample_max_points:
        sampling_max_points: int = config.alg['sampling']['max_points']
    else:
        # negative means no sampling and all points should be used
        #from the sampling_window
        sampling_max_points = -1

    print(f"sampling_max_points: {sampling_max_points}")
    # Calculate the maximum number of training points
    n_train_points = train_data_filter.filter_count_dset(porosity_data_h5)
    n_train_points = _limit_training_points(sampling_max_points,
                                            n_train_points)
    assert n_train_points > 0
    chunksize = config.alg['parallel']['max_points_per_chunk']
    train_chunkshape = _get_chunk_shape(n_train_points, chunksize)

    # Create the h5 datasets
    cur_h5, test_h5, cur_data_type = _create_files(suf_str, test_only_wells,
                                                   features_only, n_features)
    cur_h5.create_dataset(TMP_DSET_NAME, (n_train_points, ),
                          dtype=cur_data_type,
                          chunks=train_chunkshape)
    there_are_test_wells = True if len(test_only_wells) > 0 else False
    if there_are_test_wells:
        n_test_points = test_data_filter.filter_count_dset(porosity_data_h5)
        assert n_test_points > 0
        test_chunkshape = _get_chunk_shape(n_test_points, chunksize)

        test_h5.create_dataset(TMP_DSET_NAME, (n_test_points, ),
                               dtype=cur_data_type,
                               chunks=test_chunkshape)

    return (cur_h5, test_h5, sampling_max_points, train_data_filter,
            test_data_filter)


def _config_filters(test_only_wells: list, train_data_filter: DataFilter,
                    test_data_filter: DataFilter, it: int,
                    sampling_window: int) -> Tuple[DataFilter, DataFilter]:
    """
    Updates the data filters based on the presence of testing wells
    and a sampling window
    """
    there_are_test_wells = True if len(test_only_wells) > 0 else False
    if there_are_test_wells:
        train_data_filter.add_not_in_well_list_filter(test_only_wells)
        print(f"There are test wells")

    # Get config parameters and configure sampling
    if sampling_window > 0:
        train_data_filter, test_data_filter = _add_sampling_window_to_filters(
            sampling_window, train_data_filter, test_data_filter, it)

    return train_data_filter, test_data_filter


def _create_files(suf_str: str, test_only_wells: list, features_only: bool,
                  n_features: int) -> Tuple[h5py.File, h5py.File, np.dtype]:
    """
    Create the h5py files that will have the temporary datasets
    used from training and testing
    """
    filename = f'cur{suf_str}.h5'
    filename_test = f'cur{suf_str}-test.h5'

    # If the cur file exists, it should be deleted
    # A new tmp file is created by iteration
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(filename_test):
        os.remove(filename_test)

    # Creates a temporary h5 structure to maintain the porosity and features
    # data, as well as one for the test-only data, if necessary
    cur_h5 = h5py.File(f'{filename}', 'w')
    test_h5 = None
    there_are_test_wells = True if len(test_only_wells) > 0 else False
    if there_are_test_wells:
        test_h5 = h5py.File(f'{filename_test}', 'w')

    # Creates the datatype for the h5 structure, with or without 'well_id'
    if features_only:
        cur_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
        ]
    else:
        cur_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('well_id', np.int64),
        ]

    # Add the features fields and create the np datatype
    cur_data_type = cur_data_type + [(f'f{f}', np.float64)
                                     for f in range(n_features)]
    cur_data_type = np.dtype(cur_data_type)
    return cur_h5, test_h5, cur_data_type


def _get_chunk_shape(n_points: int, max_chunksize: int) -> Tuple[int]:
    """
    Doesnt expects max_chunksize to be 0. It must be checked before
    as it already is in the config_parser
    """
    chunksize = 0
    if max_chunksize < 0 or n_points < max_chunksize:
        chunksize = n_points
    else:
        chunksize = max_chunksize

    if chunksize > MAX_HDF5_CHUNK_SIZE:
        chunksize = MAX_HDF5_CHUNK_SIZE

    chunkshape = (chunksize, )
    return chunkshape


def create_tmp_dset(
    porosity_data_h5: h5py.Dataset,
    train_data_filter: DataFilter,
    n_features: int,
    config: Config,
    it: int,
    suf_str: str = '',
    list_chunk_size: int = 1000,
    features_only: bool = False,
    test_only_wells: list = None,
    should_sample_max_points: bool = True
) -> Tuple[h5py.File, h5py.Dataset, h5py.File, h5py.Dataset]:

    assert porosity_data_h5.size > 0
    if test_only_wells is None:
        test_only_wells = list()

    print(f"new test_wells: {test_only_wells}")

    profiling = False
    test_data_filter = WellsDataFilter(test_only_wells)

    t0 = time()
    (train_h5_file, test_h5_file, samp_max_points, train_data_filter,
     test_data_filter) = _prepare_h5(suf_str, test_only_wells, features_only,
                                     n_features, train_data_filter,
                                     test_data_filter, porosity_data_h5, it,
                                     config, should_sample_max_points)

    train_empty_h5_dset: h5py.Dataset = train_h5_file[TMP_DSET_NAME]
    if test_h5_file is not None:
        test_empty_h5_dset: h5py.Dataset = test_h5_file[TMP_DSET_NAME]
    else:
        test_empty_h5_dset = None

    n_training_points = train_empty_h5_dset.size

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

    (train_points_per_chunk,
     test_points_per_chunk) = _count_train_test_points_per_chunk(
         porosity_data_h5, train_data_filter, test_data_filter)

    #There will be train points for sure
    last_chunk_with_train_points = np.where(
        train_points_per_chunk > 0)[0].max()

    #There may be no test points (no test wells)
    is_there_test_points_on_c = np.where(test_points_per_chunk > 0)[0]
    if len(is_there_test_points_on_c) > 0:
        last_chunk_with_test_points = is_there_test_points_on_c.max()
    else:
        #should be 0 or negative
        last_chunk_with_test_points = 0

    must_sample = samp_max_points > 0 and n_training_points > samp_max_points
    if must_sample:
        sampling_points_per_chunk = _get_n_sampling_points_per_chunk(
            train_points_per_chunk, samp_max_points)

        rng = np.random.default_rng()

    n_test_only_wells = len(test_only_wells)
    #Go through each chunk again. Adds all test points for sure.
    for chunk_id, chunk_slice in enumerate(porosity_data_h5.iter_chunks()):
        #only reads data if necessary
        if train_points_per_chunk[chunk_id] > 0 or test_points_per_chunk[
                chunk_id] > 0:
            # Get current chunk
            chunk_np = porosity_data_h5[chunk_slice]

            # Generate test-only data, if necessary
            if test_points_per_chunk[chunk_id] > 0 and n_test_only_wells > 0:
                #Add test data to its unique list
                #could be test_data_filter.filter(chunk_np) aswell
                test_points = chunk_np[test_data_filter.satisfies(chunk_np)]

                test_feats_only = True
                test_empty_h5_dset, prev_end_test = append_points_to_dset(
                    test_feats_only, test_empty_h5_dset, prev_end_test,
                    test_points)

            if train_points_per_chunk[chunk_id] > 0:
                #could be train_data_filter.filter(chunk_np) aswell
                training_points = chunk_np[train_data_filter.satisfies(
                    chunk_np)]

                # Performs sampling on training_points
                if must_sample:
                    n_points_to_sample_chunk = sampling_points_per_chunk[
                        chunk_id]
                    training_points = _sample_points(n_points_to_sample_chunk,
                                                     rng, training_points)

                train_empty_h5_dset, prev_end = append_points_to_dset(
                    features_only, train_empty_h5_dset, prev_end,
                    training_points)

        no_more_train_chunks = chunk_id >= last_chunk_with_train_points
        no_more_test_chunks = chunk_id >= last_chunk_with_test_points
        if no_more_train_chunks and no_more_test_chunks:
            break

    t2 = time()
    if profiling:
        print(f"[get_features_sets] cur_copy_porosity_time: {t2-t1}")
        print(f"[get_features_sets] final_lenght: {train_empty_h5_dset.shape}")

    return train_h5_file, train_empty_h5_dset, test_h5_file, test_empty_h5_dset


def _sample_points(n_points_to_sample_chunk: int, rng: np.random.Generator,
                   training_points: np.ndarray) -> np.ndarray:
    """
    Sample n = min(n_points_to_sample_chunk, max_points_still_to_sample)
    points from training_points with the rng
    """

    #This accepts probabilities
    training_points = rng.choice(training_points,
                                 n_points_to_sample_chunk,
                                 replace=False)

    return training_points


def _get_n_sampling_points_per_chunk(training_points_per_chunk: np.ndarray,
                                     sampling_max_points: int) -> np.ndarray:
    """
    Calculates how many sampling points should be sampled by chunk.
    The total of sampled points is proportional to the number of training
    points in that chunk.
    The total of sampling points returned may be greater than 
    sampling_max_points so this must be checked when used.
    The total of sampling points is in the interval:
    [sampling_max_points, sampling_max_points+n_chunks]
    because of the np.ceil used
    """

    #Sample proportionally on training points per chunk
    total_train_points = np.sum(training_points_per_chunk)
    if total_train_points < sampling_max_points:
        return training_points_per_chunk.copy()

    sampling_points_per_chunk = training_points_per_chunk / total_train_points
    sampling_points_per_chunk *= sampling_max_points
    sampling_points_per_chunk = np.ceil(sampling_points_per_chunk)
    return sampling_points_per_chunk


def _count_train_test_points_per_chunk(
        porosity_data_h5: h5py.Dataset, train_data_filter: DataFilter,
        test_data_filter: DataFilter) -> Tuple[np.ndarray, np.ndarray]:
    """
    Counts how many training and testing points there are per chunk of 
    porosity_data_h5
    """

    n_chunks = _get_num_chunks_of_h5data(porosity_data_h5)

    training_points_per_chunk = np.zeros(n_chunks)
    test_points_per_chunk = np.zeros(n_chunks)

    for chunk_id, chunk_slice in enumerate(porosity_data_h5.iter_chunks()):
        chunk_np = porosity_data_h5[chunk_slice]

        n_train_p = len(train_data_filter.filter(chunk_np))
        training_points_per_chunk[chunk_id] = n_train_p

        n_test_p = len(test_data_filter.filter(chunk_np))
        test_points_per_chunk[chunk_id] = n_test_p

    return training_points_per_chunk, test_points_per_chunk


def _get_num_chunks_of_h5data(porosity_data_h5: h5py.Dataset) -> int:
    chunk_shape = np.array(porosity_data_h5.chunks)
    data_shape = np.array(porosity_data_h5.shape)
    n_chunks = int(np.prod(np.ceil(data_shape / chunk_shape)))
    return n_chunks


def append_points_to_dset(features_only: bool, target_dset: h5py.Dataset,
                          prev_end: int,
                          points: np.ndarray) -> Tuple[h5py.Dataset, int]:
    """
    Append points to the  target_dset. 
    Return the target_dset and the new prev_end
    """
    n_points = len(points)
    new_prev = prev_end + n_points
    if features_only:
        target_dset[
            'x',
            'y',
            'z',
            'phi',
            prev_end:new_prev,
        ] = points[['x', 'y', 'z', 'phi']]
    else:
        target_dset[
            'x',
            'y',
            'z',
            'phi',
            'well_id',
            prev_end:new_prev,
        ] = points[['x', 'y', 'z', 'phi', 'well_id']]

    return target_dset, new_prev


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

    data_filter = FeatSelectionTrainDataFilter()

    test_only_wells = config.alg['test_only_wells']
    wells_coords = config.wells['coords']
    training_wells = list(range(len(wells_coords)))
    training_wells = [x for x in training_wells if x not in test_only_wells]

    #At the feature selection stage, there should be sampling of
    #points from the iterations considered
    should_sample_max_points = True
    cur_h5, cur_h5_dset, test_h5, test_h5_dset = create_tmp_dset(
        porosity_data_h5,
        data_filter,
        exp_n_features,
        config,
        it,
        test_only_wells=test_only_wells,
        should_sample_max_points=should_sample_max_points)

    # Create training temporary object
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)
    cur_h5_test_list = None
    if len(test_only_wells) > 0:
        cur_h5_test_list = hdf5_util.HDFMultiColList(test_h5_dset)

    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']

    # List of features sets and their error metric
    results = []

    hypercube_shape = porosity_data_h5.shape

    # Find a feature set with exp_n_features features
    for it in range(exp_n_features):
        t3 = time()
        # Reset best feature and its error
        best_error = float('inf')
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
            print(f"[get_features_sets][{cur_feature}] "
                  f"insert_feature_time: {t5-t4}")

            # Test the model with cur_feature
            rmse, mae = eval_bootstrap(cur_h5_train_list, cur_h5_test_list,
                                       training_wells)
            t6 = time()
            print(f"[get_features_sets][{cur_feature}] "
                  f"train_time: {t6-t5}")
            print(f"[get_features_sets][{cur_feature}] "
                  f"error: {rmse}")

            results.append((cur_f_set + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_error:
                best_error = rmse
                best_feature = cur_feature

            print(f"[get_features_sets][it{it}] Tested features "
                  f"{cur_f_set+ [cur_feature]} with error {rmse}")

        # Remove the best feature from the features list
        all_features.remove(best_feature)

        t7 = time()
        print(f"[get_features_sets][it{it}] "
              f"full_it_time: {t7-t3}")

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
        print(f"[get_features_sets][{best_feature}] "
              f"commit_feature_time: {t8-t7}")

    t9 = time()
    print(f"[get_features_sets] full_time: {t9-t0}")

    cur_h5.close()
    test_h5.close()

    return get_best_features_set(results)
