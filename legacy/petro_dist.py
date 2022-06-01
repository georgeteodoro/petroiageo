from mpi4py import MPI
from enum import Enum, auto

import petro

# Defines for later externalization
LGB_MAX_THREADS = 1
MAX_FEATURES = 3
FEATURE_TEST_LIMIT = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


class MPI_TAGS(Enum):
    WORKER_EMPTY_RESULT = auto()
    MANAGER_FEATURE_DONE = auto()
    MANAGER_FINISH = auto()

# Main function to be called from outside
def get_features_sets(str_nwells):
    if rank == manager_rank:
        if mpi_size < 2:
            print("[petro-dist] 2 minimum processes required")
            return None

        best = manager(str_nwells)
        print(f"best: {best}")
        return best
    elif rank != manager_rank:
        worker(str_nwells)


def manager(str_nwells):
    print("[petro-dist][manager]")

    all_features, _ = petro.read_dataset(str_nwells)

    best_features_set = ['X', 'Y', 'depth']
    max_features = MAX_FEATURES
    feature_limiter = FEATURE_TEST_LIMIT  # 0 means no limit

    # Generate a feature set of max_features
    for f1 in range(max_features):

        # Reset workers done and waiting for next feature set
        workers_done = 0

        # Reset new best feature
        new_best_feature = None
        best_error = float("inf")

        # Create current features list as (all_features - best_features_set)
        remaining_features = [
            item for item in all_features if item not in best_features_set
        ]

        # Limit the number of features analyzed
        if feature_limiter > 0:
            remaining_features = remaining_features[:feature_limiter]

        # Iterate through all features to be tested
        while workers_done < mpi_size - 1:
            status = MPI.Status()
            data = comm.recv(status=status)
            worker_rank = status.Get_source()

            # Read results from ran feature
            if status.Get_tag() != MPI_TAGS.WORKER_EMPTY_RESULT.value:
                # Unpack data
                (cur_feature, cur_error) = data

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
        best_features_set.append(new_best_feature)

    # Broadcast a done message to all workers
    for worker_rank in range(mpi_size - 1):
        comm.send(None, dest=worker_rank, tag=MPI_TAGS.MANAGER_FINISH.value)

    return best_features_set


def worker(str_nwells):
    print(f"[petro-dist][w{rank}]")

    all_features, df = petro.read_dataset(str_nwells)

    manager_tag = None
    cur_feature_set = ['X', 'Y', 'depth']

    # Run jobs until manager finishes
    while True:
        # Request a job from manager
        comm.send(None,
                  dest=manager_rank,
                  tag=MPI_TAGS.WORKER_EMPTY_RESULT.value)

        # Get first message from Manager
        status = MPI.Status()
        new_feature = comm.recv(source=manager_rank, status=status)
        manager_tag = status.Get_tag()

        # Exit if there are no more tasks
        if manager_tag == MPI_TAGS.MANAGER_FINISH.value:
            break

        # Run jobs until there are not any
        while (manager_tag != MPI_TAGS.MANAGER_FEATURE_DONE.value
               and manager_tag != MPI_TAGS.MANAGER_FINISH.value):

            # Evaluate current features set
            rmse, mae = petro.eval_bootstrap(df,
                                             cur_feature_set + [new_feature],
                                             LGB_MAX_THREADS)

            # Return results to manager
            comm.send((new_feature, rmse), dest=manager_rank)

            # Wait for new job
            new_feature = comm.recv(source=manager_rank, status=status)
            manager_tag = status.Get_tag()

        # Get best feature from iteration from manager
        new_best_feature = comm.bcast(None, root=manager_rank)
        cur_feature_set.append(new_best_feature)


if __name__ == '__main__':
    with open("tmp_data/nwells-0.csv", mode='r') as f:
        get_features_sets(f.read())
