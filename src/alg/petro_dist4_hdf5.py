from mpi4py import MPI
from enum import Enum, auto
from time import time

import petro5_hdf5
import hdf5_util
import common
import profiling
import h5py
from typing import Dict, Tuple
from config_parser import Config
from data_filter import FeatSelectionTrainDataFilter

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


class MPI_TAGS(Enum):
    WORKER_EMPTY_RESULT = auto()  # Signals first ask from worker
    MANAGER_FEATURE_DONE = auto()  # Signals done finding new feature
    MANAGER_FINISH = auto()  # Signals done execution of current iteration
    # Signals this it should not be done anymore
    # See petro5_hdf5.eval_bootstrap comments for more
    WORKER_STOP_MSG = auto()


def get_features_sets(
    porosity_data_h5: h5py.Dataset,
    features_dict_h5: Dict[str, h5py.Dataset],
    all_features: list,
    displacement_cube_shape: tuple,
    it: int,
    config: Config,
) -> Tuple[list, float, float]:
    if mpi_size < 2:
        print("[petro4_dist_hdf5] 2 minimum processes required")
        return None

    max_feats_to_select = config.alg["max_num_features"]
    max_feats_to_test = config.get_param("max_tested_features")

    if rank == manager_rank:
        return manager(all_features, max_feats_to_select, max_feats_to_test,
                       it, config)

    else:
        return worker(
            porosity_data_h5,
            features_dict_h5,
            displacement_cube_shape,
            max_feats_to_select,
            it,
            config,
        )


def manager(
    all_features: list,
    max_feats_to_select: int,
    max_feats_to_test: int,
    it: int,
    config: Config,
) -> Tuple[list, float, float]:
    t0 = time()

    total_req_time, feats_sets_and_its_errors, t4 = _find_feats_set(
        all_features, max_feats_to_select, max_feats_to_test, it, config)

    if len(feats_sets_and_its_errors) == 0:
        best_result = None
    else:
        best_result = petro5_hdf5.get_best_features_set(
            feats_sets_and_its_errors)

    # Broadcast resulting features and errors
    comm.bcast(best_result, root=manager_rank)

    t5 = time()
    profiling.prof_fsel_manager_sync_times(it, t5 - t4, config)
    profiling.prof_fsel_manager_time(it, total_req_time + t5 - t4, t5 - t0,
                                     config)

    return best_result


def _find_feats_set(
    all_features: list,
    max_feats_to_select: int,
    max_feats_to_test: int,
    it: int,
    config: Config,
) -> Tuple[float, list[tuple], float]:
    # Profiling time counters
    total_req_time = 0
    total_sync_time = 0

    # Find a feature set by testing exp_n_features features
    curr_f_set_best_err: list[str] = ["x", "y", "z"]
    feats_sets_and_its_errors: list[tuple[list, float]] = []
    for f_it in range(max_feats_to_select):
        # Profiling time counter
        f_it_req_time = 0
        t1 = time()

        remaining_features = _remaining_feats_to_test(all_features,
                                                      curr_f_set_best_err)
        remaining_features = _limit_feats_to_test(max_feats_to_test,
                                                  remaining_features)

        f_it_req_time += time() - t1

        (new_best_feature, total_worker_time,
         curr_feats_sets) = _find_curr_best_feature(
             it,
             curr_f_set_best_err,
             remaining_features,
         )

        f_it_req_time += total_worker_time
        t3 = time()

        comm.bcast(new_best_feature, root=manager_rank)

        feats_sets_and_its_errors.extend(curr_feats_sets)
        curr_f_set_best_err.append(new_best_feature)

        t4 = time()
        total_sync_time += t4 - t3
        total_req_time += f_it_req_time
        profiling.prof_fsel_manager_sync_time(it, f_it, t4 - t3, config)
        profiling.prof_fsel_manager_req_time(it, f_it, f_it_req_time, config)

        # Means forced stop
        if new_best_feature is None:
            feats_sets_and_its_errors = []
            break

    _bcast_done_msg_to_workers()

    return total_req_time, feats_sets_and_its_errors, t4


def _bcast_done_msg_to_workers():
    for worker_rank in range(mpi_size - 1):
        # Receive EMPTY_RESULT msg to clear the queue before the next iteration
        comm.recv()
        # Actually send finish signal
        comm.send(None, dest=worker_rank, tag=MPI_TAGS.MANAGER_FINISH.value)


def _find_curr_best_feature(
    it: int,
    curr_f_set_best_err: list[str],
    remaining_features: list,
) -> Tuple[str, float, list[tuple[list, float]]]:
    new_best_feature = None
    best_rmse_error = float("inf")

    workers_done = 0
    # Iterate through all features to be tested
    total_worker_time = 0
    curr_feats_sets = list()
    forced_stop = False
    while _not_all_workers_done(workers_done):
        status = MPI.Status()
        worker_results = comm.recv(status=status)
        t2 = time()
        worker_rank = status.Get_source()

        # Number of features to be sent to the worker.
        # Currently only a single feature is sent.
        # In the future a batch of features regarding data locality
        # will be sent.
        n_features = 1

        forced_stop = _worker_forced_stop(status)
        if not forced_stop:
            if _worker_sent_feat_eval(status):
                for result in worker_results:
                    cur_feature, rmse_error, mae_error = result
                    print(f"[petro4_dist_hdf5][manager][it{it}] Tested "
                          f"feature {curr_f_set_best_err + [cur_feature]} "
                          f"with error {rmse_error}")

                    curr_feats_sets.append(
                        (curr_f_set_best_err + [cur_feature], rmse_error,
                         mae_error))

                    # Update new best, if necessary
                    if best_rmse_error > rmse_error:
                        best_rmse_error = rmse_error
                        new_best_feature = cur_feature

        # Check if should and there is work to be distributed
        if not forced_stop and len(remaining_features) > 0:
            # Send new tasks
            new_features = remaining_features[:n_features]
            remaining_features = remaining_features[n_features:]
            comm.send(new_features, dest=worker_rank)
        else:
            # Send finish message
            comm.send(None,
                      dest=worker_rank,
                      tag=MPI_TAGS.MANAGER_FEATURE_DONE.value)
            workers_done = workers_done + 1

        total_worker_time += time() - t2

    if forced_stop:
        new_best_feature = None
        curr_feats_sets = []

    return new_best_feature, total_worker_time, curr_feats_sets


def _worker_sent_feat_eval(status: MPI.Status) -> bool:
    return status.Get_tag() != MPI_TAGS.WORKER_EMPTY_RESULT.value


def _worker_forced_stop(status: MPI.Status) -> bool:
    return status.Get_tag() == MPI_TAGS.WORKER_STOP_MSG.value


def _not_all_workers_done(workers_done):
    return workers_done < mpi_size - 1


def _limit_feats_to_test(max_feats_to_test, remaining_features):
    if max_feats_to_test > 0:
        remaining_features = remaining_features[:max_feats_to_test]
    return remaining_features


def _remaining_feats_to_test(all_features, curr_f_set_best_err) -> list:
    return [item for item in all_features if item not in curr_f_set_best_err]


def worker(
    porosity_data_h5: h5py.Dataset,
    features_dict_h5: Dict[str, h5py.Dataset],
    displacement_cube_shape: tuple,
    exp_n_features: int,
    it: int,
    config: Config,
) -> Tuple[list, float, float]:
    t0 = time()

    data_filter = FeatSelectionTrainDataFilter()

    test_wells_ids = config.alg["test_only_wells"]

    #At the feature selection stage, there should be sampling of
    #points from the iterations considered
    cur_h5, cur_h5_dset, _, _ = petro5_hdf5.create_tmp_dset(
        porosity_data_h5,
        data_filter,
        exp_n_features,
        config,
        it,
        f"-r{rank}",
        test_wells_ids=test_wells_ids,
        should_sample_max_points=True,
        generate_test_files=False)

    hypercube_shape: tuple = porosity_data_h5.shape

    # Create training temporary object
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)

    t1 = time()
    profiling.prof_fsel_worker_create_time(it, rank, t1 - t0, config)

    t8, total_jobs, total_exec_time = _eval_feats_requested_by_manager(
        features_dict_h5,
        displacement_cube_shape,
        it,
        config,
        cur_h5_dset,
        hypercube_shape,
        cur_h5_train_list,
    )
    cur_h5.close()

    profiling.prof_fsel_worker_times(it, rank, total_exec_time, t8 - t0,
                                     total_jobs, config)

    # Get broadcasted resulting features and errors
    best_result = comm.bcast(None, root=manager_rank)
    return best_result


def _eval_feats_requested_by_manager(
    features_dict_h5: Dict[str, h5py.Dataset],
    displacement_cube_shape: tuple,
    it: int,
    config: Config,
    cur_h5_dset: h5py.Dataset,
    hypercube_shape: tuple,
    cur_h5_train_list: hdf5_util.HDFMultiColList,
) -> Tuple[float, int, float]:
    cur_f_set = ["x", "y", "z"]

    # Profiling info
    total_jobs = 0
    total_exec_time = 0
    while True:
        # For profiling
        f_it = len(cur_h5_train_list.all_features) + 1
        t2 = time()

        status = MPI.Status()
        # Request a job from manager
        comm.send(None,
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)
        new_features, manager_tag = _get_new_feats_from_manager(status)

        t3 = time()
        profiling.prof_fsel_worker_comm_time(it, rank, t3 - t2, config)

        if _all_expected_feats_sets_tested(manager_tag):
            break

        # Setup the new column to be tested
        cur_h5_train_list.add_new_col()

        while _there_are_feats_to_test(manager_tag):
            results, total_jobs, curr_total_time = _eval_curr_feats(
                features_dict_h5,
                displacement_cube_shape,
                it,
                config,
                cur_h5_dset,
                hypercube_shape,
                cur_h5_train_list,
                total_jobs,
                f_it,
                new_features,
            )
            total_exec_time += curr_total_time

            t6 = time()

            # Return results to manager
            if results is None:
                comm.send(results,
                          dest=manager_rank,
                          status=MPI_TAGS.WORKER_STOP_MSG.value)
            else:
                comm.send(results, dest=manager_rank)

            new_features, manager_tag = _get_new_feats_from_manager(status)

            t7 = time()
            profiling.prof_fsel_worker_comm_time(it, rank, t7 - t6, config)

        t7 = time()

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_f_set.append(new_best_feature)

        # Insert best selected feature
        petro5_hdf5.insert_filtered_feature(
            cur_h5_dset,
            cur_h5_train_list,
            features_dict_h5,
            new_best_feature,
            hypercube_shape,
            displacement_cube_shape,
        )

        t8 = time()
        profiling.prof_fsel_worker_sync_time(it, rank, f_it, t8 - t7, config)

    return t8, total_jobs, total_exec_time


def _eval_curr_feats(
    features_dict_h5: Dict[str, h5py.Dataset],
    displacement_cube_shape: tuple,
    it: int,
    config: Config,
    cur_h5_dset: h5py.Dataset,
    hypercube_shape: tuple,
    cur_h5_train_list: hdf5_util.HDFMultiColList,
    total_jobs: int,
    f_it: int,
    new_features: list[str],
) -> Tuple[list[Tuple[str, float, float]], int, float]:
    results: list[Tuple[str, float, float]] = []
    train_wells_ids = config.train_wells_ids
    total_time = 0
    for new_feature in new_features:
        t4 = time()

        # Insert temporary feature
        petro5_hdf5.insert_filtered_feature(
            cur_h5_dset,
            cur_h5_train_list,
            features_dict_h5,
            new_feature,
            hypercube_shape,
            displacement_cube_shape,
        )

        t5 = time()
        profiling.prof_fsel_worker_insert_time(it, rank, f_it, t5 - t4, config)
        rmse, mae = petro5_hdf5.eval_bootstrap(cur_h5_train_list,
                                               train_wells_ids)

        if (rmse, mae) == (None, None):
            results = None
            return results, total_jobs, total_time
        else:
            results.append((new_feature, rmse, mae))
            t6 = time()
            profiling.prof_fsel_worker_eval_times(it, rank, f_it, t6 - t5,
                                                  config)

            total_jobs += 1
            total_time += t6 - t4

    return results, total_jobs, total_time


def _there_are_feats_to_test(manager_tag) -> bool:
    return manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value


def _all_expected_feats_sets_tested(manager_tag: int) -> bool:
    return manager_tag == MPI_TAGS.MANAGER_FINISH.value


def _get_new_feats_from_manager(status: MPI.Status) -> Tuple[list[str], int]:
    new_features: list[str] = comm.recv(source=manager_rank, status=status)
    manager_tag = status.Get_tag()
    return new_features, manager_tag


if __name__ == "__main__":
    with open("tmp_data/nwells-0.csv", mode="r") as f:
        get_features_sets(f.read())
