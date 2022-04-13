from mpi4py import MPI
from enum import Enum, auto
import time

import petro2

# Initialization of mpi variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


class MPI_TAGS(Enum):
    WORKER_EMPTY_RESULT = auto()  # Signals first ask from worker
    MANAGER_FEATURE_DONE = auto()  # Signals done finding new feature
    MANAGER_FINISH = auto()  # Signals done execution of current iteration


# exp_n_features: number of features to be selected
# f_width: number of features to be compared
#   default=0 means all features.
#   Used for debugging and reducing computing cost
def get_features_sets(main_df,
                      features_df,
                      all_features,
                      exp_n_features,
                      f_width=0):
    if mpi_size < 2:
        print("[petro-dist] 2 minimum processes required")
        return None

    if rank == manager_rank:
        return manager(all_features, exp_n_features, f_width)
    elif rank != manager_rank:
        return worker(main_df, features_df)


def manager(all_features, exp_n_features, f_width):
    print("[petro-dist][manager]")

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

        # Limit the number of features analyzed
        if f_width > 0:
            remaining_features = remaining_features[:f_width]

        # Iterate through all features to be tested
        while workers_done < mpi_size - 1:
            status = MPI.Status()
            data = comm.recv(status=status)
            worker_rank = status.Get_source()

            # Read results from ran feature
            if status.Get_tag() != MPI_TAGS.WORKER_EMPTY_RESULT.value:
                # Unpack data
                (cur_feature, cur_error) = data

                print(f'Tested feature {cur_f_set + [cur_feature]} '\
                      f'with error {cur_error}')

                results.append((cur_f_set + [cur_feature], cur_error))

                # Update new best, if necessary
                if best_error > cur_error:
                    best_error = cur_error
                    new_best_feature = cur_feature

            # Check if there is work to be distributed
            if len(remaining_features) > 0:
                # Send new task
                new_feature = remaining_features.pop()
                comm.send(new_feature, dest=worker_rank)
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
        print(f'[petro2] fullIt time: {t1-t0}')

    # Broadcast a done message to all workers
    for worker_rank in range(mpi_size - 1):
        # Receive EMPTY_RESULT msg to clear the queue before the next iteration
        comm.recv()
        # Actually send finish signal
        comm.send(None, dest=worker_rank, tag=MPI_TAGS.MANAGER_FINISH.value)

    # Broadcast resulting features and errors
    best_result = petro2.get_best_features_set(results)
    comm.bcast(best_result, root=manager_rank)

    return best_result


def worker(main_df, features_df):
    print(f"[petro-dist][w{rank}]")

    # all_features, df = petro.read_dataset(str_nwells)

    # Create a shallow copy of main_df for adding new columns
    # Data from is main_df is only referenced, not copied
    cur_df = main_df.copy(deep=False)

    cur_f_set = ['x', 'y', 'z']

    # Run jobs until manager finishes
    while True:
        # Request a job from manager
        comm.send(None,
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)

        avg_feature_time = 0
        avg_feature_time_count = 0

        # Get first message from Manager
        status = MPI.Status()
        new_feature = comm.recv(source=manager_rank, status=status)
        manager_tag = status.Get_tag()

        # Exit if there are no more tasks
        if manager_tag == MPI_TAGS.MANAGER_FINISH.value:
            break

        # Run jobs until there are not any
        print(f"[petro-dist][w{rank}] new iteration")
        while (manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value):

            print(f"[petro-dist][w{rank}] testing {cur_f_set + [new_feature]}")
            t1 = time.time()
            rmse, mae = petro2.single_feature_run(cur_df, features_df,
                                                  new_feature)
            t2 = time.time()
            avg_feature_time = avg_feature_time + (t2 - t1)
            avg_feature_time_count = avg_feature_time_count + 1

            # Return results to manager
            comm.send((new_feature, rmse), dest=manager_rank)

            # Wait for new job
            new_feature = comm.recv(source=manager_rank, status=status)
            manager_tag = status.Get_tag()

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_f_set.append(new_best_feature)
        cur_df.loc[:,
                   petro2.f2str(new_best_feature)] = petro2.get_feature_col2(
                       cur_df.index, new_best_feature, features_df)

        if avg_feature_time_count > 0:
            print(f'[petro-dist][w{rank}][profiling] {avg_feature_time_count}'\
                  f' tasks ran with avg feature time: '\
                  f'{avg_feature_time/avg_feature_time_count}')
        else:
            print('[petro-dist][w{rank}][profiling] 0 tasks ran '\
                  'with avg feature time: 0')

    # Get broadcasted resulting features and errors
    best_result = comm.bcast(None, root=manager_rank)
    return best_result


if __name__ == '__main__':
    with open("tmp_data/nwells-0.csv", mode='r') as f:
        get_features_sets(f.read())
