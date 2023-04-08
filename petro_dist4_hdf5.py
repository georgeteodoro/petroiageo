from mpi4py import MPI
from enum import Enum, auto
import time
import concurrent.futures
import ctypes
import multiprocessing as mp
import sys

import petro5_hdf5
import hdf5_util
import common

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


class MPI_TAGS(Enum):
    WORKER_EMPTY_RESULT = auto()  # Signals first ask from worker
    MANAGER_FEATURE_DONE = auto()  # Signals done finding new feature
    MANAGER_FINISH = auto()  # Signals done execution of current iteration


# # exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(
        porosity_data_h5,
        features_dict_h5,
        all_features,
        window_sizes,
        displacement_cube_shape,
        # parallel_settings,
        it_str,
        exp_n_features,
        f_width=0):

    # main_ddf,
    # features_ddf,
    # all_features,
    # hypercube_shape,
    # dask_chunksize,
    # parallel_settings,
    # exp_n_features,
    # f_width=0):

    if mpi_size < 2:
        print("[petro5_hdf5] 2 minimum processes required")
        return None

    if rank == manager_rank:
        return manager(all_features, exp_n_features, f_width, it_str)
    elif rank != manager_rank:
        # return worker(main_ddf, features_ddf, hypercube_shape, dask_chunksize,
        #               parallel_settings)
        return worker(porosity_data_h5, features_dict_h5, window_sizes,
                      displacement_cube_shape, exp_n_features, it_str)


def manager(all_features, exp_n_features, f_width, it_str):
    # print("[petro5_hdf5][manager]")

    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']

    # List of features sets and their error metric
    results = []

    # Find a feature set by testing exp_n_features features
    for _ in range(exp_n_features):

        t0 = time.time()

        # Reset workers done and wait for next feature set
        workers_done = 0

        # Reset new best feature
        new_best_feature = None
        best_error = float("inf")

        # Create current features list as (all_features - cur_f_set)
        remaining_features = [
            item for item in all_features if item not in cur_f_set
        ]

        # print(f'======================= [manager] got '\
        #       f'remaining_features[:10]: {remaining_features[:10]}')

        # Limit the number of features analyzed
        if f_width > 0:
            remaining_features = remaining_features[:f_width]
        # print(f'==========1=============')

        # Iterate through all features to be tested
        while workers_done < mpi_size - 1:
            # print(f'==========2=============')
            status = MPI.Status()
            # print(f'==========3=============')
            data = comm.recv(status=status)
            # print(f'======================= [manager] got worker msg {data}')
            worker_rank = status.Get_source()

            # Number of features to be sent to the worker.
            # Currently only a single feature is sent.
            # In the future a batch of features regarding data locality
            # will be sent.
            n_features = 1

            # Read results from ran feature
            if status.Get_tag() != MPI_TAGS.WORKER_EMPTY_RESULT.value:
                for (cur_feature, cur_error) in data:
                    print(f'[petro5_hdf5][manager]{it_str} Tested feature '\
                          f'{cur_f_set + [cur_feature]} '\
                          f'with error {cur_error}')

                    results.append((cur_f_set + [cur_feature], cur_error))

                    # Update new best, if necessary
                    if best_error > cur_error:
                        best_error = cur_error
                        new_best_feature = cur_feature

            # Check if there is work to be distributed
            if len(remaining_features) > 0:
                # Send new tasks
                new_features = remaining_features[:n_features]
                remaining_features = remaining_features[n_features:]
                # print(f'======================= [manager] '\
                #       f'new_features: {new_features}')
                comm.send(new_features, dest=worker_rank)
            else:
                # Send finish message
                comm.send(None,
                          dest=worker_rank,
                          tag=MPI_TAGS.MANAGER_FEATURE_DONE.value)
                workers_done = workers_done + 1

        # Broadcast new best feature and updates current best features_set
        comm.bcast(new_best_feature, root=manager_rank)
        cur_f_set.append(new_best_feature)

        t1 = time.time()
        print(f'[petro5_hdf5][manager]{it_str} fullIt time: {t1-t0}')

    # Broadcast a done message to all workers
    for worker_rank in range(mpi_size - 1):
        # Receive EMPTY_RESULT msg to clear the queue before the next iteration
        comm.recv()
        # Actually send finish signal
        comm.send(None, dest=worker_rank, tag=MPI_TAGS.MANAGER_FINISH.value)

    # Broadcast resulting features and errors
    best_result = petro5_hdf5.get_best_features_set(results)
    comm.bcast(best_result, root=manager_rank)

    return best_result


def worker(porosity_data_h5,
           features_dict_h5,
           window_sizes,
           displacement_cube_shape,
           exp_n_features,
           it_str,
           parallel_settings=None):
    # print(f"[petro5_hdf5][w{rank}]")

    # Points used for training: real, expanded and propagated
    is_training_point_f = lambda d: (
        (d['real'] == common.RealValues.real) |
        (d['real'] == common.RealValues.canal_expanded) |
        (d['real'] == common.RealValues.expanded) |
        (d['real'] == common.RealValues.propagated))

    cur_h5, cur_h5_dset = petro5_hdf5.create_tmp_dset(porosity_data_h5,
                                                      is_training_point_f,
                                                      exp_n_features,
                                                      f'-r{rank}')

    hypercube_shape = porosity_data_h5.shape

    # Create training temporary object
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)

    cur_f_set = ['x', 'y', 'z']

    # Run jobs until manager finishes
    while True:
        t0 = time.time()

        # Request a job from manager
        comm.send(None,
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)

        # print(f'======================= [worker] sent mpi WORKER_EMPTY_RESULT')

        # Profiling info
        total_feature_exec_time = 0
        total_feature_comm_time = 0
        feature_exec_count = 0

        # Get first message from Manager
        status = MPI.Status()
        # print('======================= [worker] waiting recv from manager')
        new_features = comm.recv(source=manager_rank, status=status)
        manager_tag = status.Get_tag()

        # print(
        #     f'======================= [worker] got mpi job msg {new_features}')

        # Exit if there are no more tasks (all expected features sets were tested)
        if manager_tag == MPI_TAGS.MANAGER_FINISH.value:
            break

        # Setup the new column to be tested
        cur_h5_train_list.add_new_col()

        # Run jobs until there are not any more features to test
        print(f'[petro5_hdf5][w{rank}]{it_str} new iteration')
        while (manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value):
            print(f'[petro5_hdf5][w{rank}]{it_str} Received new_features: '\
                  f'{new_features}')

            # Run all features received by the manager
            results = []
            for new_feature in new_features:
                print(f'[petro5_hdf5][w{rank}]{it_str} Testing feature: '\
                      f'{new_feature}')
                t1 = time.time()
                # Insert temporary feature
                petro5_hdf5.insert_filtered_feature(cur_h5_dset,
                                                    cur_h5_train_list,
                                                    features_dict_h5,
                                                    new_feature, window_sizes,
                                                    hypercube_shape,
                                                    displacement_cube_shape)

                rmse, mae = petro5_hdf5.eval_bootstrap(cur_h5_train_list,
                                                       list(range(10)))

                results.append((new_feature, rmse))
                t2 = time.time()
                total_feature_exec_time = total_feature_exec_time + (t2 - t1)

            # Return results to manager
            comm.send(results, dest=manager_rank)

            # Wait for new job
            new_features = comm.recv(source=manager_rank, status=status)
            manager_tag = status.Get_tag()

            t3 = time.time()
            total_feature_comm_time = total_feature_comm_time + (t3 - t2)
            feature_exec_count = feature_exec_count + 1

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_f_set.append(new_best_feature)

        # Insert best selected feature
        petro5_hdf5.insert_filtered_feature(cur_h5_dset, cur_h5_train_list,
                                            features_dict_h5, new_best_feature,
                                            window_sizes, hypercube_shape,
                                            displacement_cube_shape)

        t4 = time.time()

        print(
            f'[petro5_hdf5][w{rank}][profiling]{it_str} it_full_time: {t4-t0}')
        print(f'[petro5_hdf5][w{rank}][profiling]{it_str} total_exec_time: '\
              f'{total_feature_exec_time}')
        print(f'[petro5_hdf5][w{rank}][profiling]{it_str} total_comm_time: '\
              f'{total_feature_comm_time}')
        print(f'[petro5_hdf5][w{rank}][profiling]{it_str} n_tasks: '\
              f'{feature_exec_count}')

    # Get broadcasted resulting features and errors
    best_result = comm.bcast(None, root=manager_rank)
    return best_result


if __name__ == '__main__':
    with open("tmp_data/nwells-0.csv", mode='r') as f:
        get_features_sets(f.read())
