from mpi_module import MPI_TAGS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

beg_str = f"[worker{rank}]"


def run(config):
    seismic_on_mem = config.get()
    td_on_mem = config.get()
    should_update = self.config.get_param("mpi_should_update_local")

    # Prep data
    seismic_dict = _load_seismic()
    test_data = _pepare_test_data()

    status = MPI.Status()

    best_features = ['x', 'y', 'z']

    for it in range(nits):
        # Update test data: set test_data size and update coordinates,
        # porosity, and other columns
        test_data.prepare_porosity(it)

        # feature selection
        comm.send(None, dest=manager_rank, tag=MPI_TAGS.WORKER_FIRST_JOB.value)

        while True:
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()

            # Respond the received job
            if msg_tag == MPI_TAGS.MANAGER_NEW_JOB.value:
                # Got new feature to analyze
                new_feature = msg
                test_data.update_feature(new_feature)
                ret = _test_new_feature(test_data, config)

                # Send response back
                comm.send(None,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_JOB_RESULT.value)

            elif msg_tag == MPI_TAGS.MANAGER_SELECTED_FEATURE.value:
                # Got the best feature for a f_it
                new_feature = msg
                best_features.append(new_feature)
                test_data.update_feature(new_feature)
                test_data.commit_feature()

            elif msg_tag == MPI_TAGS.MANAGER_BEST_FEATURES.value:
                # Generate the best features list
                # best_features = ['x', 'y', 'z']
                best_features = []
                best_features += msg

                # Add the last column to test_data
                test_data.update_feature(best_features[-1])

                break

            else:
                raise Exception(f"[manager] Bad MPI tag: {msg_tag}")

        # propagation
        if should_update:
            propagate(it, best_features, config)
