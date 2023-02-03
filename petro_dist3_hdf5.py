from mpi4py import MPI
from enum import Enum, auto
import time
import concurrent.futures
import ctypes
import multiprocessing as mp
import sys

import petro4_hdf5
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
        print("[petro-dist] 2 minimum processes required")
        return None

    if rank == manager_rank:
        return manager(all_features, exp_n_features, f_width)
    elif rank != manager_rank:
        # return worker(main_ddf, features_ddf, hypercube_shape, dask_chunksize,
        #               parallel_settings)
        return worker(porosity_data_h5, features_dict_h5,
                      displacement_cube_shape, exp_n_features)


def manager(all_features, exp_n_features, f_width):
    # print("[petro-dist][manager]")

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

            # Number of features to be sent to the worker
            n_features = -1

            # Read results from ran feature
            if status.Get_tag() != MPI_TAGS.WORKER_EMPTY_RESULT.value:
                # Unpack data
                (data, n_features) = data
                for (cur_feature, cur_error) in data:
                    print(f'[petro-dist][manager]{it_str} Tested feature '\
                          f'{cur_f_set + [cur_feature]} '\
                          f'with error {cur_error}')

                    results.append((cur_f_set + [cur_feature], cur_error))

                    # Update new best, if necessary
                    if best_error > cur_error:
                        best_error = cur_error
                        new_best_feature = cur_feature
            else:
                # First msg only sends the number of requested features
                n_features = data

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
        print(f'[petro3]{it_str} fullIt time: {t1-t0}')

    # Broadcast a done message to all workers
    for worker_rank in range(mpi_size - 1):
        # Receive EMPTY_RESULT msg to clear the queue before the next iteration
        comm.recv()
        # Actually send finish signal
        comm.send(None, dest=worker_rank, tag=MPI_TAGS.MANAGER_FINISH.value)

    # Broadcast resulting features and errors
    best_result = petro4_hdf5.get_best_features_set(results)
    comm.bcast(best_result, root=manager_rank)

    return best_result


# # Wrapper to get shared variables
# def single_feature_run_proxy(feature, hypercube_shape, dask_chunksize, n_cpu):
#     return petro3.single_feature_run(cur_ddf_shr.value, features_ddf_shr.value,
#                                      feature, hypercube_shape, dask_chunksize,
#                                      n_cpu)


def worker(porosity_data_h5,
           features_dict_h5,
           displacement_cube_shape,
           exp_n_features,
           parallel_settings=None):
    # print(f"[petro-dist][w{rank}]")

    if not parallel_settings is None:
        n_cpus = parallel_settings['n_cpus']
        cpu_thrds = parallel_settings['cpu_thrds']
    else:
        n_cpus = 1
        cpu_thrds = 1

    # # Create a shallow copy of main_ddf for adding new columns
    # # Data from is main_ddf is only referenced, not copied
    # cur_ddf = main_ddf.copy()
    # print('======================= [worker] done main_ddf.copy()')

    # Points used for training: real, expanded and propagated
    is_training_point_f = lambda d: (
        (d['real'] == common.RealValues.real) |
        (d['real'] == common.RealValues.canal_expanded) |
        (d['real'] == common.RealValues.expanded) |
        (d['real'] == common.RealValues.propagated))

    cur_h5, cur_h5_dset = petro4_hdf5.create_tmp_dset(porosity_data_h5,
                                                      is_training_point_f,
                                                      exp_n_features,
                                                      f'-r{rank}')

    hypercube_shape = porosity_data_h5.shape

    # Create sequence object
    cur_h5_seq = hdf5_util.HDFMultiColSequence(cur_h5_dset, ['x', 'y', 'z'])

    cur_f_set = ['x', 'y', 'z']

    # Run jobs until manager finishes
    while True:
        t0 = time.time()

        # Request a job from manager
        comm.send(n_cpus,
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)

        # print(f'======================= [worker] sent mpi WORKER_EMPTY_RESULT')

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
        cur_h5_seq.add_new_col()

        # Run jobs until there are not any more features to test
        print(f"[petro-dist][w{rank}]{it_str} new iteration")
        while (manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value):
            if n_cpus == 1:
                t1 = time.time()
                # Insert temporary feature
                petro4_hdf5.insert_filtered_feature(cur_h5_dset, cur_h5_seq,
                                                    features_dict_h5,
                                                    new_features[0],
                                                    hypercube_shape,
                                                    displacement_cube_shape)

                rmse, mae = petro4_hdf5.eval_bootstrap(cur_h5_seq,
                                                       list(range(10)))

                # (rmse, mae) = single_feature_run(
                #     cur_ddf, features_ddf, new_features[0], hypercube_shape,
                #     dask_chunksize, cpu_thrds)

                # # Get future results
                # results = [f.result() for f in future]
                # results = [rmse for (rmse, mae) in results]

                t2 = time.time()

                # Return results to manager
                comm.send(([(new_features[0], rmse)], n_cpus),
                          dest=manager_rank)
            else:
                raise Exception('not implemented...')
                t1 = time.time()
                print(f'[petro-dist][w{rank}]{it_str} executing '\
                      f'{len(new_features)} features in parallel')
                with concurrent.futures.ThreadPoolExecutor(n_cpus) as executor:
                    future = [
                        executor.submit(single_feature_run_proxy, f,
                                        hypercube_shape, dask_chunksize,
                                        cpu_thrds) for f in new_features
                    ]

                # Get future results
                results = [f.result() for f in future]
                results = [rmse for (rmse, mae) in results]

                t2 = time.time()
                print(f'[petro-dist][w{rank}]{it_str} ran {len(new_features)} '\
                      f'features in parallel in {t2-t1} secs')

                # Return results to manager
                comm.send((list(zip(new_features, results)), n_cpus),
                          dest=manager_rank)

            # Wait for new job
            new_feature = comm.recv(source=manager_rank, status=status)
            manager_tag = status.Get_tag()

            t3 = time.time()
            total_feature_exec_time = total_feature_exec_time + (t2 - t1)
            total_feature_comm_time = total_feature_comm_time + (t3 - t2)
            feature_exec_count = feature_exec_count + 1

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_f_set.append(new_best_feature)

        # Insert best selected feature
        petro4_hdf5.insert_filtered_feature(cur_h5_dset, cur_h5_seq,
                                            features_dict_h5, new_features[0],
                                            hypercube_shape,
                                            displacement_cube_shape)

        # cur_ddf[petro3.f2str(new_best_feature)] = petro3.get_feature_col2(
        #     cur_ddf.index, new_best_feature, features_ddf)
        # cur_ddf = cur_ddf.persist()

        t4 = time.time()

        print(
            f'[petro-dist][w{rank}][profiling]{it_str} it_full_time: {t4-t0}')
        print(f'[petro-dist][w{rank}][profiling]{it_str} total_exec_time: '\
              f'{total_feature_exec_time}')
        print(f'[petro-dist][w{rank}][profiling]{it_str} total_comm_time: '\
              f'{total_feature_comm_time}')
        print(f'[petro-dist][w{rank}][profiling]{it_str} n_tasks: '\
              f'{feature_exec_count}')

    # Get broadcasted resulting features and errors
    best_result = comm.bcast(None, root=manager_rank)
    return best_result


if __name__ == '__main__':
    with open("tmp_data/nwells-0.csv", mode='r') as f:
        get_features_sets(f.read())
