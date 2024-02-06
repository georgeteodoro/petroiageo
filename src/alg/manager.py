from mpi4py import MPI

from mpi_module import MPI_TAGS
import FeatureDataBase

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
workers_size = mpi_size - 1
manager_rank = mpi_size - 1

beg_str = "[manager]"

# Manager only works on feature selection. It does not performs propagation


def _send_new_features(remaining_features, worker_rank, config):
    '''
    Send a batch of features to a worker.
    Currently this is somewhat empty, but later scheduling code will
    be put here.
    New features are returned just for debugging purposes.
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

    comm.send(new_features,
              dest=worker_rank,
              tag=MPI_TAGS.MANAGER_NEW_JOB.value)

    return new_features


def run(config):
    # Get config parameters
    n_features_to_select = config.alg['max_num_features']
    max_feats_to_test = config.get_param("max_tested_features")
    num_its = config.alg['num_its']
    start_it = config.alg['it']

    status = MPI.Status()

    # Count aborted workers. If any worker aborts, all workers should
    # abort.
    aborted_workers = 0

    cur_best_feature = None
    cur_best_rmse = float('inf')
    cur_best_mae = float('inf')

    for it in range(start_it, num_its + start_it):
        print(beg_str + f" Running [it{it}]")

        # Prepare features lists
        all_features = FeatureDataBase.gen_features_list(config)
        remaining_features = all_features.copy()
        if max_feats_to_test > 0:
            remaining_features = remaining_features[:max_feats_to_test]
        best_features = []

        # Count of how many workers are just waiting the end of
        # the current f_it. It only changes when there are no more
        # remaining_features.
        done_workers = 0

        # Main loop on which a whole iteration is run
        while True:
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()
            worker_rank = status.Get_source()

            # Parse response from worker
            if msg_tag == MPI_TAGS.WORKER_JOB_RESULT.value:
                # Update best feature, if new best was found
                for (feature, rmse, mae) in msg:
                    if rmse < cur_best_rmse:
                        cur_best_rmse = rmse
                        cur_best_mae = mae
                        cur_best_feature = feature
            elif msg_tag == MPI_TAGS.WORKER_FIRST_JOB.value:
                # Currently nothing to do on this case
                pass

            elif msg_tag == MPI_TAGS.WORKER_ABORT_PROP.value:
                print(beg_str + f"[it{it}] Received abort from {worker_rank}.")

                # After first abort signal, there is nothing else to do
                # with the current worker.
                aborted_workers += 1
                if aborted_workers == workers_size:
                    return

                continue

            else:
                raise Exception(f"{beg_str}[it{it}] Bad MPI tag: {msg_tag}")

            # If one worker has aborted, there is nothing else to do besides
            # sending an abort signal to the current worker and wait for
            if aborted_workers > 0:
                print(beg_str + f"[it{it}] Sending abort to w{worker_rank}.")

                aborted_workers += 1
                comm.send(None,
                          dest=worker_rank,
                          tag=MPI_TAGS.MANAGER_ABORT_PROP.value)

                # If all workers have aborted, then manager can abort too
                if aborted_workers == workers_size:
                    return

                continue

            if len(remaining_features) > 0:
                # There are still features to test on this f_it
                new_features = _send_new_features(remaining_features,
                                                  worker_rank, config)
                print(beg_str + f"[it{it}] Sending features {new_features}")
            else:
                done_workers += 1
                # If all workers are done, then this is the end of a f_it
                # or a full iteration
                if done_workers == workers_size:
                    best_features.append(cur_best_feature)

                    # Check if this is the final f_it from the current it,
                    # or if this is just the end of a f_it.
                    if len(best_features) < n_features_to_select:
                        # End the current f_it

                        # Send best current feature to all workers
                        for worker_rank in range(workers_size):
                            print(
                                beg_str +
                                f"[it{it}] New best feature {cur_best_feature}")
                            comm.send(
                                cur_best_feature,
                                dest=worker_rank,
                                tag=MPI_TAGS.MANAGER_SELECTED_FEATURE.value)

                        # Reload new remaining features without the
                        # chosen feature
                        all_features.remove(cur_best_feature)
                        remaining_features = all_features.copy()
                        if max_feats_to_test > 0:
                            remaining_features = remaining_features[:
                                                                    max_feats_to_test]

                        # Reset temporary variables
                        done_workers = 0
                        cur_best_feature = None
                        cur_best_rmse = float('inf')
                        cur_best_mae = float('inf')
                    else:
                        # End the current it
                        best_feat_string = ""
                        for feat_name, feat_disp in best_features:
                            best_feat_string += feat_name + " " + " ".join(
                                [str(disp) for disp in feat_disp]) + ","
                        best_feat_string = best_feat_string.rstrip(",")
                        best_feat_string = "[" + best_feat_string + "]"
                        print(
                            beg_str +
                            f"[it{it}] Iteration best features: {best_feat_string}"
                            +
                            f" with errors: MAE {cur_best_mae} RMSE {cur_best_rmse}"
                        )

                        # Send best features set to all workers
                        for worker_rank in range(workers_size):
                            print(
                                beg_str +
                                f"[it{it}] Sending final best features to worker{worker_rank}"
                            )
                            comm.send(best_features,
                                      dest=worker_rank,
                                      tag=MPI_TAGS.MANAGER_BEST_FEATURES.value)
                        break
