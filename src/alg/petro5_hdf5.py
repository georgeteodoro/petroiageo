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
        features_sets: list[tuple[list, float]]) -> Tuple[list, float, float]:
    # Sort by second column (id 1)
    features_sets.sort(key=lambda tup: tup[1])
    best_features_set = features_sets[0][0]
    rmse_error = features_sets[0][1]
    mae_error = features_sets[0][2]

    return best_features_set, rmse_error, mae_error


def eval_bootstrap(
    cur_h5_train_list: hdf5_util.HDFMultiColList,
    wells_id: list[int],
    num_threads=1,
) -> Tuple[float, float]:
    """
    Train a regressor on cur_h5_train_list using data of wells in wells_id.
    The test errors will be based on the validation data for each 
    Leave-one-well-out iteration.
    
    Return the mean rmse and mean mae errors. May return None, None
    """
    params['num_threads'] = num_threads

    profiling = False

    rmse_list, mae_list = _leave_one_well_out_training(cur_h5_train_list,
                                                       wells_id, profiling)

    # See _leave_one_well_out_training comments to understand when this happens
    if len(rmse_list) == 0 or len(mae_list) == 0:
        return None, None

    return np.mean(rmse_list), np.mean(mae_list)


def _leave_one_well_out_training(
    cur_h5_train_list: hdf5_util.HDFMultiColList,
    wells_id: int,
    profiling: bool,
) -> Tuple[list[float], list[float]]:

    t0 = time()
    rmse_list: list[float] = []
    mae_list: list[float] = []

    # Leave-One-Well-Out (LOWO)
    for curr_well_id in wells_id:
        X_val, y_val, trained_regressor = _incremental_learning(
            cur_h5_train_list, profiling, curr_well_id)

        # This happens when there are no points associated
        # with well_id in the last its_window_size iterations
        #  as said in _incremental_learning. So we just ignore
        if X_val is not None:
            # This happens when all points in the last its_window_size
            # iterations are associated with only one well.
            # As X_val is not None, that well is the validation well
            # at this it of LOWO and there were no remaining training points
            # so the regressor was None. At this point, there is no need
            # to continue to try the training so we return
            if trained_regressor is None:
                return [], []

            t3 = time()

            # Calculate error metrics

            pred = trained_regressor.predict(X_val)
            rmse = np.sqrt(np.mean((pred - y_val)**2))
            mae = mean_absolute_error(pred, y_val)
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

    # When using its_window_size in the config, there will, probably,
    # be some iteration where there were no points sampled from some
    # well because it stopped having new points propagated its_window_size
    # iterations ago. For example, when a well is totally
    # surrounded by other wells. So we ignore this validation
    if len(X_val) == 0 | len(y_val) == 0:
        return None, None, None

    # Incremental training on all cur_h5_train_list chunks
    regressor = None
    setup_time = 0
    training_time = 0
    # TODO: Check train data availability on chunk without loading data
    for chunk_idx in range(cur_h5_train_list.n_chunks):
        t1 = time()

        X_train, y_train = cur_h5_train_list.get_data_not_in_well(
            chunk_idx, curr_well_id)
        # We might be in a chunk full of data of well chunk_idx
        if len(X_train) > 0:
            lgb_train_dataset = lgb.Dataset(X_train, y_train)

            lgb_eval_dataset = lgb.Dataset(X_val,
                                           y_val,
                                           reference=lgb_train_dataset)

            t2 = time()
            regressor = lgb.train(
                params,
                lgb_train_dataset,
                init_model=regressor,
                num_boost_round=100,
                valid_sets=lgb_eval_dataset,
                keep_training_booster=True,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False)
                ],
            )
            t3 = time()

            setup_time += t2 - t1
            training_time += t3 - t2

            if profiling:
                print(
                    f"[petro5_hdf5][eval_bootstrap][w{curr_well_id}] Setup in "
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
                                    it) -> DataFilter:
    """
    Updates the train filter based on sampling_window
    """
    min_ring = max(0, it - sampling_window)

    train_data_filter.add_min_ring_filter(min_ring)

    return train_data_filter


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
                config: Config,
                generate_test_file: bool,
                should_sample_max_points: bool = True):
    """
    Generate the h5 File and dataset objects.
    Also returns the total number of sampled training points. If no sampling is
    done, returns all training and test points.
    If should_sample_max_points, then considers the max_points param of the 
    sampling param in config. If not, then no sampling of max points
    is used
    """

    #We may not have the sampling window and still have
    #sampling max points defined
    if should_sample_max_points:
        sampling_max_points: int = config.alg['sampling']['max_points']
    else:
        # negative means no sampling and all points should be used
        #from the sampling_window
        sampling_max_points = -1

    # Calculate the maximum number of training points
    n_train_points = train_data_filter.filter_count_dset(porosity_data_h5)
    n_train_points = _limit_training_points(sampling_max_points,
                                            n_train_points)
    assert n_train_points > 0
    chunksize = config.alg['parallel']['max_points_per_chunk']
    train_chunkshape = _get_chunk_shape(n_train_points, chunksize)

    # Create the h5 datasets
    cur_h5, test_h5, cur_data_type = _create_files(suf_str, test_only_wells,
                                                   features_only, n_features,
                                                   generate_test_file)
    cur_h5.create_dataset(TMP_DSET_NAME, (n_train_points, ),
                          dtype=cur_data_type,
                          chunks=train_chunkshape)
    there_are_test_wells = True if len(test_only_wells) > 0 else False
    if there_are_test_wells and generate_test_file:
        n_test_points = test_data_filter.filter_count_dset(porosity_data_h5)
        assert n_test_points > 0
        test_chunkshape = _get_chunk_shape(n_test_points, chunksize)

        test_h5.create_dataset(TMP_DSET_NAME, (n_test_points, ),
                               dtype=cur_data_type,
                               chunks=test_chunkshape)

    return (cur_h5, test_h5, sampling_max_points)


def _config_filters(test_wells_ids: list, train_data_filter: DataFilter,
                    it: int,
                    sampling_window: int) -> Tuple[DataFilter, DataFilter]:
    """
    Updates the train data filter based on the presence of testing wells
    and a sampling window. The test data filter is not updated here
    because it must have only it's real points. If we were to add the sampling
    window filter to the test data, at some point, the real points would
    be outside the n size sampling range.
    """
    there_are_test_wells = True if len(test_wells_ids) > 0 else False
    if there_are_test_wells:
        train_data_filter.add_not_in_well_list_filter(test_wells_ids)

    # Get config parameters and configure sampling
    if sampling_window > 0:
        train_data_filter = _add_sampling_window_to_filters(
            sampling_window, train_data_filter, it)

    return train_data_filter


def _create_files(
        suf_str: str, test_only_wells: list, features_only: bool,
        n_features: int,
        generate_test_file: bool) -> Tuple[h5py.File, h5py.File, np.dtype]:
    """
    Create the h5py files that will have the temporary datasets
    used from training and testing
    """
    filename = f'cur{suf_str}.h5'

    # If the cur file exists, it should be deleted
    # A new tmp file is created by iteration
    if os.path.exists(filename):
        os.remove(filename)

    # Creates a temporary h5 structure to maintain the porosity and features
    # data, as well as one for the test-only data, if necessary
    cur_h5 = h5py.File(f'{filename}', 'w')

    test_h5 = None
    there_are_test_wells = True if len(test_only_wells) > 0 else False
    if there_are_test_wells and generate_test_file:
        filename_test = f'cur{suf_str}-test.h5'
        if os.path.exists(filename_test):
            os.remove(filename_test)
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
    features_only: bool = False,
    test_wells_ids: list = None,
    should_sample_max_points: bool = True,
    generate_test_files: bool = False
) -> Tuple[h5py.File, h5py.Dataset, h5py.File, h5py.Dataset]:

    assert porosity_data_h5.size > 0
    if test_wells_ids is None:
        test_wells_ids = list()

    profiling = False
    test_data_filter = WellsDataFilter(test_wells_ids)

    train_data_filter = _config_filters(
        test_wells_ids, train_data_filter, it,
        config.alg['sampling']['its_window_size'])

    t0 = time()
    (train_h5_file, test_h5_file, samp_max_points) = _prepare_h5(
        suf_str, test_wells_ids, features_only, n_features, train_data_filter,
        test_data_filter, porosity_data_h5, config, generate_test_files,
        should_sample_max_points)

    train_empty_h5_dset: h5py.Dataset = train_h5_file[TMP_DSET_NAME]
    if test_h5_file is not None:
        test_empty_h5_dset: h5py.Dataset = test_h5_file[TMP_DSET_NAME]
    else:
        test_empty_h5_dset = None

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
         porosity_data_h5,
         train_data_filter,
         test_data_filter,
         there_are_test_data=generate_test_files)

    total_n_training_points_possible = np.sum(train_points_per_chunk)

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

    must_sample = samp_max_points > 0 and total_n_training_points_possible > samp_max_points
    if must_sample:
        sampling_points_per_chunk = _get_n_sampling_points_per_chunk(
            train_points_per_chunk, samp_max_points)

    rng = np.random.default_rng(seed=config.alg['sampling']['seed'])

    n_test_only_wells = len(test_wells_ids)
    #Go through each chunk again. Adds all test points for sure.
    for chunk_id, chunk_slice in enumerate(porosity_data_h5.iter_chunks()):
        #only reads data if necessary
        if train_points_per_chunk[chunk_id] > 0 or test_points_per_chunk[
                chunk_id] > 0:
            # Get current chunk
            chunk_np = porosity_data_h5[chunk_slice]

            # Generate test-only data, if necessary
            # There is only going to be test points in some chunk if
            # generate_test_files is True
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

                assert np.all(training_points["well_id"] >= 0)
                # Performs sampling on training_points
                if must_sample:
                    n_points_to_sample_chunk = sampling_points_per_chunk[
                        chunk_id]
                    training_points = _sample_points_from_chunk(
                        n_points_to_sample_chunk, rng, training_points)

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


def _sample_points_from_chunk(n_points_to_sample_chunk: int,
                              rng: np.random.Generator,
                              training_points: np.ndarray) -> np.ndarray:
    """
    Sample n_points_to_sample_chunk points from training_points with the rng. 
    It sample points from every well in the chunk proportionally.

    Example: 
    If n_points_to_sample_chunk=100 and there are 3 wells
    [1,2,3] with [100, 200, 300] points respectivelly in the chunk,
    then, it will try to sample [17, 33, 50] points respectivelly
    """

    ordered_well_ids, well_ids_count = np.unique(training_points['well_id'],
                                                 return_counts=True)

    n_samp_points_per_well = _get_n_pts_to_sample_per_well(
        n_points_to_sample_chunk, well_ids_count)

    assert_msg = "Num of sampled points of some well was bellow 1!"
    assert_msg += f" {n_samp_points_per_well}"
    assert len(
        n_samp_points_per_well[n_samp_points_per_well <= 0]) == 0, assert_msg

    sampled_training_points = None
    for well_idx, well_id in enumerate(ordered_well_ids):
        n_samp_points_well = n_samp_points_per_well[well_idx]
        well_points = training_points[training_points['well_id'] == well_id]
        assert np.all(well_points['well_id'] == well_id)
        #This accepts probabilities
        curr_sampled_points = rng.choice(well_points,
                                         n_samp_points_well,
                                         replace=False)

        if sampled_training_points is None:
            sampled_training_points = curr_sampled_points
        else:
            sampled_training_points = np.concatenate(
                [sampled_training_points, curr_sampled_points])

    assert sampled_training_points.size >= n_points_to_sample_chunk

    return sampled_training_points


def _get_n_pts_to_sample_per_well(n_points_to_sample_chunk: int,
                                  n_pts_per_well: np.ndarray) -> np.ndarray:
    # This might give more points to sample in total than
    # n_points_to_sample_chunk because of the ceil. So it must be treated
    n_samp_points_per_well: np.ndarray = np.ceil(
        (n_pts_per_well / n_pts_per_well.sum()) *
        n_points_to_sample_chunk).astype(int)

    # Treating difference to expected n_points_to_sample_chunk
    # We remove 1 point from every well with biggest curr samp size
    # until diff == 0. This next diff will never be < 0.
    diff = n_samp_points_per_well.sum() - n_points_to_sample_chunk

    while diff > 0:
        biggest_samp = n_samp_points_per_well.argmax()
        n_samp_points_per_well[biggest_samp] -= 1
        diff = n_samp_points_per_well.sum() - n_points_to_sample_chunk

    # This garantees that every well has at least one training point at the end
    n_samp_points_per_well = np.where(n_samp_points_per_well <= 0, 1,
                                      n_samp_points_per_well)

    return n_samp_points_per_well


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
    sampling_points_per_chunk = np.ceil(sampling_points_per_chunk).astype(int)
    return sampling_points_per_chunk


def _count_train_test_points_per_chunk(
        porosity_data_h5: h5py.Dataset,
        train_data_filter: DataFilter,
        test_data_filter: DataFilter,
        there_are_test_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Counts how many training and testing points there are per chunk of 
    porosity_data_h5
    """

    n_chunks = _get_num_chunks_of_h5data(porosity_data_h5)

    training_points_per_chunk = np.zeros(n_chunks)
    test_points_per_chunk = np.zeros(n_chunks)

    for chunk_id, chunk_slice in enumerate(porosity_data_h5.iter_chunks()):
        chunk_np = porosity_data_h5[chunk_slice]

        train_filtered = train_data_filter.filter(chunk_np)
        assert np.all((train_filtered['well_id'] >= 0))
        n_train_p = len(train_filtered)
        training_points_per_chunk[chunk_id] = n_train_p

        if there_are_test_data:
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

    # As the sampling_points may pass the target_dset.size
    # (see _get_n_sampling_points_per_chunk)
    # we must check if the curr n_points would lead to a
    # shape error on target_dset
    diff = new_prev - target_dset.size
    if diff > 0:
        new_prev = target_dset.size
        points = points[:-diff]

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


def get_features_sets(
    porosity_data_h5: h5py.Dataset,
    features_dict_h5: Dict[str, h5py.Dataset],
    all_features: list,
    displacement_cube_shape: tuple,
    alg_it: int,
    config: Config,
) -> Tuple[list, float, float]:
    """
    This method evals all feature combinations on train data and return the
    best one with its errors
    Args:
        porosity_data_h5: The dataset
        features_dict_h5: A dict with the features data
        all_features: The features names to test
        displacement_cube_shape: 
        alg_it: The current algorithm total iteration
        config: The Config dict

    Returns: 
        Tuple (best_feats_combination, rmse_error, mae_error)
    """
    t0 = time()

    data_filter = FeatSelectionTrainDataFilter()

    test_wells_ids = config.alg['test_only_wells']
    train_wells_ids = config.train_wells_ids

    #At the feature selection stage, there should be sampling of
    #points from the iterations considered
    exp_n_features = config.alg["max_num_features"]
    cur_h5, cur_h5_dset, _, _ = create_tmp_dset(porosity_data_h5,
                                                data_filter,
                                                exp_n_features,
                                                config,
                                                alg_it,
                                                test_wells_ids=test_wells_ids,
                                                should_sample_max_points=True,
                                                generate_test_files=False)

    # Create training temporary object
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)

    # Current features set with the best error
    best_feats_found = ['x', 'y', 'z']

    # List of features sets and their error metric
    # We must save EVERY combination
    results = []

    hypercube_shape = porosity_data_h5.shape
    max_tested_features = config.get_param("max_tested_features")

    # Find a feature set with exp_n_features features
    for feat_sel_it in range(exp_n_features):
        t3 = time()
        # Reset best feature and its error
        best_rmse_error = float('inf')
        curr_best_feature = None

        # Setup the new column to be tested
        cur_h5_train_list.add_new_col()

        # Test each available feature
        for feat_idx, cur_feature in enumerate(all_features):

            if cur_feature in best_feats_found:
                continue

            t4 = time()
            # Early termination for debugging
            if max_tested_features != 0 and feat_idx == max_tested_features:
                break

            # Insert a temporary feature
            insert_filtered_feature(
                cur_h5_dset,
                cur_h5_train_list,
                features_dict_h5,
                cur_feature,
                hypercube_shape,
                displacement_cube_shape,
            )

            t5 = time()
            print(f"[get_features_sets][{cur_feature}] "
                  f"insert_feature_time: {t5-t4}")

            # Test the model with cur_feature
            rmse, mae = eval_bootstrap(cur_h5_train_list, train_wells_ids)

            t6 = time()
            print(f"[get_features_sets][{cur_feature}] train_time: {t6-t5}")

            if (rmse, mae) == (None, None):
                results = None
                curr_best_feature = None
                break

            print(f"[get_features_sets][{cur_feature}] RMSE: {rmse}")

            results.append((best_feats_found + [cur_feature], rmse, mae))

            # Update current best feature
            if rmse < best_rmse_error:
                best_rmse_error = rmse
                curr_best_feature = cur_feature

            msg = f"[get_features_sets][it{alg_it}][feat_sel_it{feat_sel_it}]"
            msg += f"Tested features {best_feats_found+ [cur_feature]}"
            msg += f" with error: RMSE {rmse}; MAE {mae}"
            print(msg)

        t7 = time()
        print(f"[get_features_sets][it{alg_it}][feat_sel_it{feat_sel_it}] "
              f"full_it_time: {t7-t3}")

        if curr_best_feature is None:
            break

        # Update the last column of the sequence object to the best feature
        insert_filtered_feature(
            cur_h5_dset,
            cur_h5_train_list,
            features_dict_h5,
            curr_best_feature,
            hypercube_shape,
            displacement_cube_shape,
        )

        # Mark feature as already selected
        best_feats_found.append(curr_best_feature)

        t8 = time()
        print(f"[get_features_sets][{curr_best_feature}] "
              f"commit_feature_time: {t8-t7}")

    t9 = time()
    print(f"[get_features_sets] full_time: {t9-t0}")

    cur_h5.close()

    return None if results is None else get_best_features_set(results)
