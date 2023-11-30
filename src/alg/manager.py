from mpi4py import MPI

from mpi_module import MPI_TAGS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

beg_str = "[manager] "

# Manager only works on feature selection. It does not performs propagation


def _gen_features_list(config):
    num_features = config.get_param('num_features')
    base_features = config.features_files_names
    window_size = config.alg['window']

    # Ignore any other file which is not an .h5 file.
    base_features = [f for f in base_features if f != ".gitkeep"]
    assert len(base_features) > 0, "No features found."

    # Limit features list to the maximum size
    if num_features > 0:
        base_features = base_features[:num_features]

    # Expand features for all displacements
    all_features = []
    for f in base_features:
        for i in range(-window_size, window_size + 1):
            for j in range(-window_size, window_size + 1):
                for k in range(-window_size, window_size + 1):
                    all_features.append((f, i, j, k))

    return all_features


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
    new_features = [
        remaining_features.pop() for i in range(batch_features_size)
    ]

    print(beg_str + f"Sending features {new_features}")
    comm.send(new_features,
              dest=worker_rank,
              tag=MPI_TAGS.MANAGER_NEW_JOB.value)


def run(config):
    # Get config parameters
    n_features_to_select = config.alg['max_num_features']
    max_feats_to_test = config.get_param("max_tested_features")
    num_its = config.alg['num_its']
    start_it = config.alg['it']

    # Prepare features lists
    all_features = _gen_features_list(config)
    remaining_features = all_features.copy()
    if max_feats_to_test > 0:
        remaining_features = remaining_features[:max_feats_to_test]
    best_features = []

    status = MPI.Status()

    # Count of how many workers are just waiting the end of
    # the current f_it. It only changes when there are no more
    # remaining_features.
    done_workers = 0

    cur_best_feature = None
    cur_best_metric = float('inf')

    for it in range(start_it, num_its + start_it):
        print(beg_str + f"Running it[{it}]")
        # Main loop on which a whole iteration is run
        while True:
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()
            worker_rank = status.Get_source()

            # Parse response from worker
            if msg_tag == MPI_TAGS.WORKER_JOB_RESULT.value:
                # Update best feature, if new best was found
                print(msg)
                for (feature, rmse, mae) in msg:
                    if rmse < cur_best_metric:
                        cur_best_metric = rmse
                        cur_best_feature = feature
            elif msg_tag == MPI_TAGS.WORKER_FIRST_JOB.value:
                # Currently nothing to do on this case
                pass
            else:
                raise Exception(f"{beg_str} Bad MPI tag: {msg_tag}")

            if len(remaining_features) > 0:
                # There are still features to test on this f_it
                _send_new_features(remaining_features, worker_rank, config)
            else:
                done_workers += 1
                # If all workers are done, then this is the end of a f_it
                # or a full iteration
                if done_workers == mpi_size - 1:
                    best_features.append(cur_best_feature)

                    # Check if this is the final f_it from the current it,
                    # or if this is just the end of a f_it.
                    if len(best_features) < n_features_to_select:
                        # End the current f_it

                        # Send best current feature to all workers
                        for worker_rank in range(mpi_size - 1):
                            print(beg_str +
                                  f"New best feature {cur_best_feature}")
                            comm.send(
                                cur_best_feature,
                                dest=worker_rank,
                                tag=MPI_TAGS.MANAGER_SELECTED_FEATURE.value)

                        # Reload new remaining features
                        all_features.remove(cur_best_feature)
                        remaining_features = all_features.copy()
                        if max_feats_to_test > 0:
                            remaining_features = remaining_features[:max_feats_to_test]

                        # Reset temporary variables
                        done_workers = 0
                        cur_best_feature = None
                        cur_best_metric = float('inf')
                    else:
                        # End the current it

                        # Send best features set to all workers
                        for worker_rank in range(mpi_size - 1):
                            print(
                                beg_str +
                                f"Sending final best features {best_features}")
                            comm.send(best_features,
                                      dest=worker_rank,
                                      tag=MPI_TAGS.MANAGER_BEST_FEATURES.value)
                        break
