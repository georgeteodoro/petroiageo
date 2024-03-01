from mpi4py import MPI

from mpi_module import MPI_TAGS
from FeatureSchedFIFO import FeatureSchedFIFO
from FeatureSchedFLoc import FeatureSchedFLoc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
workers_size = mpi_size - 1
manager_rank = mpi_size - 1

beg_str = "[manager]"

# Manager only works on feature selection. It does not performs propagation


def run(config):
    # Get config parameters
    n_features_to_select = config.alg['max_num_features']
    feat_loc_scheduler = config.get_param('fsched_loc')
    num_its = config.alg['num_its']
    start_it = config.alg['it']

    status = MPI.Status()

    # Count aborted workers. If any worker aborts, all workers should
    # abort.
    aborted_workers = 0

    if feat_loc_scheduler:
        feature_scheduler = FeatureSchedFLoc(config)
    else:
        feature_scheduler = FeatureSchedFIFO(config)

    cur_best_feature = None
    cur_best_rmse = float('inf')
    cur_best_mae = float('inf')

    for it in range(start_it, num_its + start_it):
        print(f"{beg_str} Running [it{it}]")

        # Initialize best features and local features to be scheduled
        best_features = []
        feature_scheduler.begin_iteration()

        # Count of how many workers are just waiting the end of
        # the current f_it. It only changes when there are no more
        # features to be scheduled.
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
                    feature_scheduler.tried_feature(feature, worker_rank)
                    if rmse < cur_best_rmse:
                        cur_best_rmse = rmse
                        cur_best_mae = mae
                        cur_best_feature = feature
            elif msg_tag == MPI_TAGS.WORKER_FIRST_JOB.value:
                # Currently nothing to do on this case
                pass

            elif msg_tag == MPI_TAGS.WORKER_ABORT_PROP.value:
                print(f"{beg_str}[it{it}] Received abort from {worker_rank}.")

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
                print(f"{beg_str}[it{it}] Sending abort to w{worker_rank}.")

                aborted_workers += 1
                comm.send(None,
                          dest=worker_rank,
                          tag=MPI_TAGS.MANAGER_ABORT_PROP.value)

                # If all workers have aborted, then manager can abort too
                if aborted_workers == workers_size:
                    return

                continue

            new_feature = feature_scheduler.get_feature(worker_rank)
            if new_feature is not None:
                # There are still features to test on this f_it
                # The list encapsulation is to later enable batching
                comm.send([new_feature],
                          dest=worker_rank,
                          tag=MPI_TAGS.MANAGER_NEW_JOB.value)
                print(f"{beg_str}[it{it}] Sending feature {new_feature}")
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
                            print(f"{beg_str}[it{it}] New best feature "
                                  f"{cur_best_feature}")
                            comm.send(
                                cur_best_feature,
                                dest=worker_rank,
                                tag=MPI_TAGS.MANAGER_SELECTED_FEATURE.value)

                        # Reload new remaining features without the
                        # chosen feature
                        feature_scheduler.commit_feature(cur_best_feature)

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
                        print(f"{beg_str}[it{it}] Iteration best features: "
                              f"{best_feat_string} with errors: "
                              f"MAE {cur_best_mae} RMSE {cur_best_rmse}")

                        # Send best features set to all workers
                        for worker_rank in range(workers_size):
                            print(f"{beg_str}[it{it}] Sending final best "
                                  f"features to worker{worker_rank}")
                            comm.send(best_features,
                                      dest=worker_rank,
                                      tag=MPI_TAGS.MANAGER_BEST_FEATURES.value)

                        # Wait all workers to be at the end of it after possibly
                        # propagating
                        end_of_it_workers = 0
                        while True:
                            msg = comm.recv(status=status)
                            msg_tag = status.Get_tag()
                            worker_rank = status.Get_source()
                            if msg_tag == MPI_TAGS.WORKER_END_OF_IT.value:
                                end_of_it_workers += 1
                            else:
                                raise Exception(
                                    f"{beg_str}[it{it}] Manager expected {MPI_TAGS.WORKER_END_OF_IT.value} but got {msg_tag}"
                                )

                            if end_of_it_workers == workers_size:
                                for worker_rank in range(workers_size):
                                    comm.send([],
                                              dest=worker_rank,
                                              tag=MPI_TAGS.
                                              MANAGER_LIBERATE_WORKERS.value)
                                break
                        break
