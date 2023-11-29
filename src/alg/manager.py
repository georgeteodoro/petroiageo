from tqdm import tqdm

# Manager only works on feature selection. It does not performs propagation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1


class MPI_TAGS(Enum):
    # First job request from a worker. Can be issued at the beginning of
    # the iteration or after a best feature is found and received.
    WORKER_FIRST_JOB = auto()

    # Training results sent by a worker, also requesting a new job.
    WORKER_JOB_RESULT = auto()

    # New feature to be evaluated by a worker, send by the manager
    MANAGER_NEW_JOB = auto()

    # Best feature from the current f_it, sent by the manager. This message
    # implies that more features should be tested, thus, should be followed
    # by a WORKER_FIRST_JOB message from all workers.
    MANAGER_SELECTED_FEATURE = auto()

    # Best features set of the current iteration. This means that the feature
    # selection stage is done for the current iteration.
    MANAGER_BEST_FEATURES = auto()


def _update_best_feature(cur_best_feature, cur_best_metric, msg, config):
    '''
    This is the place for adjusting the selection of error metrics 
    for features.
    '''

    


def _send_new_features(remaining_features, worker_rank, config):
    '''
    Send a batch of features to a worker.
    Currently this is somewhat empty, but later scheduling code will
    be put here.
    '''

    # Number of features to be sent to the worker.
    # Currently only a single feature is sent.
    # In the future a batch of features regarding data locality
    # will be sent.
    batch_features_size = 1
    # batch_features_size = config.get()

    # Pop a batch of features
    new_features = remaining_features[:batch_features_size]
    remaining_features = remaining_features[batch_features_size:]

    comm.send(new_features,
              dest=worker_rank,
              tag=MPI_TAGS.MANAGER_NEW_JOB.value)


def run(config):
    all_features = [...]
    remaining_features = all_features.copy()
    best_features = []
    n_features_to_select = config.get()

    # Count of how many workers are just waiting the end of
    # the current f_it. It only changes when there are no more
    # remaining_features.
    done_workers = 0

    cur_best_feature = None
    cur_best_metric = None

    for it in range(its):
        # Main loop on which a whole iteration is run
        while True:
            status = MPI.Status()
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()
            worker_rank = status.Get_source()

            # Parse response from worker
            if msg_tag == WORKER_JOB_RESULT:
                # Update best feature, if new best was found
                for (feature, rmse, mae) in msg:
                    if rmse < cur_best_metric:
                        cur_best_metric = rmse
                        cur_best_feature = feature
            elif msg_tag == WORKER_FIRST_JOB:
                # Currently nothing to do on this case
                pass
            else:
                raise Exception(f"[manager] Bad MPI tag: {msg_tag}")

            if len(remaining_features) > 0:
                # There are still features to test on this f_it
                _send_new_features(remaining_features, worker_rank, config)
            else:
                done_workers += 1
                # If all workers are done, then this is the end of a f_it
                # or a full iteration
                if done_workers == mpi_size:
                    best_features.append(cur_best_feature)

                    # Check if this is the final f_it from the current it,
                    # or if this is just the end of a f_it.
                    if len(best_features) < n_features_to_select:
                        # End the current f_it

                        # Send best current feature to all workers
                        for worker_rank in range(mpi_size - 1):
                            comm.send(cur_best_feature,
                                      dest=worker_rank,
                                      tag=MANAGER_SELECTED_FEATURE)

                        # Reload new remaining features
                        all_features.remove(cur_best_feature)
                        remaining_features = all_features.copy()

                        # Reset temporary variables
                        done_workers = 0
                        cur_best_feature = None
                        cur_best_metric = None
                    else:
                        # End the current it

                        # Send best current feature to all workers
                        for worker_rank in range(mpi_size - 1):
                            comm.send(best_features,
                                      dest=worker_rank,
                                      tag=MANAGER_BEST_FEATURES)
                        break
