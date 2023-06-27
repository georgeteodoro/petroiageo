from mpi4py import MPI
from enum import Enum, auto
from time import time

import petro5_hdf5
import hdf5_util
import common
import profiling

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
def get_features_sets(porosity_data_h5, features_dict_h5, all_features,
                      displacement_cube_shape, it, exp_n_features, f_width,
                      config):

    if mpi_size < 2:
        print("[petro4_dist_hdf5] 2 minimum processes required")
        return None

    if rank == manager_rank:
        return manager(all_features, exp_n_features, f_width, it, config)
    elif rank != manager_rank:
        return worker(porosity_data_h5, features_dict_h5,
                      displacement_cube_shape, exp_n_features, it, config)


def manager(all_features, exp_n_features, f_width, it, config):
    # Current features set with the best error
    cur_f_set = ['x', 'y', 'z']

    # List of features sets and their error metric
    results = []

    # Profiling time counters
    total_req_time = 0
    total_sync_time = 0

    t0 = time()

    # Find a feature set by testing exp_n_features features
    for f_it in range(exp_n_features):

        # Profiling time counter
        f_it_req_time = 0

        t1 = time()

        # Reset workers done and wait for next feature set
        workers_done = 0

        # Reset new best feature
        new_best_feature = None
        best_error = float("inf")

        # Create current features list as (all_features - cur_f_set)
        remaining_features = [
            item for item in all_features if item not in cur_f_set
        ]

        # Limit the number of features analyzed
        if f_width > 0:
            remaining_features = remaining_features[:f_width]

        f_it_req_time += time() - t1

        # Iterate through all features to be tested
        while workers_done < mpi_size - 1:
            status = MPI.Status()
            data = comm.recv(status=status)
            t2 = time()
            worker_rank = status.Get_source()

            # Number of features to be sent to the worker.
            # Currently only a single feature is sent.
            # In the future a batch of features regarding data locality
            # will be sent.
            n_features = 1

            # Read results from ran feature
            if status.Get_tag() != MPI_TAGS.WORKER_EMPTY_RESULT.value:
                for (cur_feature, cur_error) in data:
                    print(f'[petro4_dist_hdf5][manager][it{it}] Tested '\
                          f'feature {cur_f_set + [cur_feature]} '\
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
                comm.send(new_features, dest=worker_rank)
            else:
                # Send finish message
                comm.send(None,
                          dest=worker_rank,
                          tag=MPI_TAGS.MANAGER_FEATURE_DONE.value)
                workers_done = workers_done + 1

            t3 = time()
            f_it_req_time += t3 - t2

        # Broadcast new best feature and updates current best features_set
        comm.bcast(new_best_feature, root=manager_rank)
        cur_f_set.append(new_best_feature)

        t4 = time()
        total_sync_time += t4 - t3
        total_req_time += f_it_req_time
        profiling.prof_fsel_manager_sync_time(it, f_it, t4 - t3, config)
        profiling.prof_fsel_manager_req_time(it, f_it, f_it_req_time, config)

    # Broadcast a done message to all workers
    for worker_rank in range(mpi_size - 1):
        # Receive EMPTY_RESULT msg to clear the queue before the next iteration
        comm.recv()
        # Actually send finish signal
        comm.send(None, dest=worker_rank, tag=MPI_TAGS.MANAGER_FINISH.value)

    # Broadcast resulting features and errors
    best_result = petro5_hdf5.get_best_features_set(results)
    comm.bcast(best_result, root=manager_rank)

    t5 = time()
    profiling.prof_fsel_manager_sync_times(it, t5 - t4, config)
    profiling.prof_fsel_manager_time(it, total_req_time + t5 - t4, t5 - t0,
                                     config)

    return best_result


def worker(porosity_data_h5, features_dict_h5, displacement_cube_shape,
           exp_n_features, it, config):

    t0 = time()

    # Points used for training: real, expanded and propagated
    is_training_point_f = lambda d: (
        (d['real'] == common.RealValues.real) |
        (d['real'] == common.RealValues.canal_expanded) |
        (d['real'] == common.RealValues.propagated))

    test_only_wells = config.alg['test_only_wells']
    wells_coords = config.wells['coords']
    training_coords = list(range(len(wells_coords)))
    training_coords = [x for x in training_coords if x not in test_only_wells]

    cur_h5, cur_h5_dset, test_h5, test_h5_dset = petro5_hdf5.create_tmp_dset(
        porosity_data_h5,
        is_training_point_f,
        exp_n_features,
        f'-r{rank}',
        test_only_wells=test_only_wells)

    hypercube_shape = porosity_data_h5.shape

    # Create training temporary object
    cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)
    cur_h5_test_list = None
    if len(test_only_wells) > 0:
        cur_h5_test_list = hdf5_util.HDFMultiColList(test_h5_dset)

    t1 = time()
    profiling.prof_fsel_worker_create_time(it, rank, t1 - t0, config)

    cur_f_set = ['x', 'y', 'z']

    # Profiling info
    total_jobs = 0
    total_exec_time = 0

    # Run jobs until manager finishes
    while True:
        # For profiling
        f_it = len(cur_h5_train_list.all_features) + 1

        t2 = time()

        # Request a job from manager
        comm.send(None,
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)

        # Get first message from Manager
        status = MPI.Status()
        new_features = comm.recv(source=manager_rank, status=status)
        manager_tag = status.Get_tag()

        t3 = time()
        profiling.prof_fsel_worker_comm_time(it, rank, t3 - t2, config)

        # Exit if there are no more tasks (all expected features sets were tested)
        if manager_tag == MPI_TAGS.MANAGER_FINISH.value:
            break

        # Setup the new column to be tested
        cur_h5_train_list.add_new_col()
        if len(test_only_wells) > 0:
            cur_h5_test_list.add_new_col()

        # Run jobs until there are not any more features to test
        # print(f'[petro4_dist_hdf5][w{rank}][it{it}] new iteration')
        while (manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value):
            # print(f'[petro4_dist_hdf5][w{rank}][it{it}] '\
            #       f'Received new_features: {new_features}')

            # Run all features received by the manager
            results = []
            for new_feature in new_features:
                # print(f'[petro4_dist_hdf5][w{rank}][it{it}] '\
                #       f'Testing feature: {new_feature}')
                t4 = time()
                # Insert temporary feature
                petro5_hdf5.insert_filtered_feature(
                    cur_h5_dset, cur_h5_train_list, features_dict_h5,
                    new_feature, hypercube_shape, displacement_cube_shape)

                # Also inserts the feature on the test dataset, if necessary
                if len(test_only_wells) > 0:
                    petro5_hdf5.insert_filtered_feature(
                        test_h5_dset, cur_h5_test_list, features_dict_h5,
                        new_feature, hypercube_shape, displacement_cube_shape)

                t5 = time()
                profiling.prof_fsel_worker_insert_time(it, rank, f_it, t5 - t4,
                                                       config)

                rmse, mae = petro5_hdf5.eval_bootstrap(cur_h5_train_list,
                                                       cur_h5_test_list,
                                                       training_coords)

                results.append((new_feature, rmse))
                t6 = time()
                profiling.prof_fsel_worker_eval_times(it, rank, f_it, t6 - t5,
                                                      config)

                total_jobs += 1
                total_exec_time += t6 - t4

            t6 = time()

            # Return results to manager
            comm.send(results, dest=manager_rank)

            # Wait for new job
            new_features = comm.recv(source=manager_rank, status=status)
            manager_tag = status.Get_tag()

            t7 = time()
            profiling.prof_fsel_worker_comm_time(it, rank, t7 - t6, config)

        t7 = time()

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_f_set.append(new_best_feature)

        # Insert best selected feature
        petro5_hdf5.insert_filtered_feature(cur_h5_dset, cur_h5_train_list,
                                            features_dict_h5, new_best_feature,
                                            hypercube_shape,
                                            displacement_cube_shape)

        # Also inserts the feature on the test dataset, if necessary
        if len(test_only_wells) > 0:
            petro5_hdf5.insert_filtered_feature(test_h5_dset, cur_h5_test_list,
                                                features_dict_h5,
                                                new_best_feature,
                                                hypercube_shape,
                                                displacement_cube_shape)

        t8 = time()
        profiling.prof_fsel_worker_sync_time(it, rank, f_it, t8 - t7, config)

    profiling.prof_fsel_worker_times(it, rank, total_exec_time, t8 - t0,
                                     total_jobs, config)

    # Get broadcasted resulting features and errors
    best_result = comm.bcast(None, root=manager_rank)
    return best_result


if __name__ == '__main__':
    with open("tmp_data/nwells-0.csv", mode='r') as f:
        get_features_sets(f.read())
