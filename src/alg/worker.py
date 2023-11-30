from mpi4py import MPI

from mpi_module import MPI_TAGS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

beg_str = f"[worker{rank}] "


def run(config):
    seismic_on_mem = config.alg['seismic_on_memory']
    td_on_mem = config.alg['TD_on_memory']
    fsel_only = config.alg['fsel_only']
    should_update = config.get_param("mpi_should_update_local")
    num_its = config.alg['num_its']

    # Prep data
    # seismic_dict = _load_seismic()
    # test_data = _pepare_test_data()

    status = MPI.Status()

    for it in range(num_its):
        best_features = ['x', 'y', 'z']

        # Update test data: set test_data size and update coordinates,
        # porosity, and other columns
        # test_data.prepare_porosity(it)

        # feature selection
        comm.send(None, dest=manager_rank, tag=MPI_TAGS.WORKER_FIRST_JOB.value)

        while True:
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()

            # Respond the received job
            if msg_tag == MPI_TAGS.MANAGER_NEW_JOB.value:
                print(beg_str + f"Got new feature to test {msg}")

                # Got new feature to analyze
                new_features = msg
                results = []
                for feature in new_features:
                    # test_data.update_feature(new_feature)
                    # ret = _test_new_feature(test_data, config)
                    ret = (2, 1)
                    results.append((feature, *ret))

                # Send response back
                comm.send(results,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_JOB_RESULT.value)

            elif msg_tag == MPI_TAGS.MANAGER_SELECTED_FEATURE.value:
                print(beg_str + f"Got new best feature {msg}")
                # Got the best feature for a f_it
                new_feature = msg
                best_features.append(new_feature)
                # test_data.update_feature(new_feature)
                # test_data.commit_feature()

                # Send response back requesting new job
                comm.send(results,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_FIRST_JOB.value)

            elif msg_tag == MPI_TAGS.MANAGER_BEST_FEATURES.value:
                print(beg_str + f"Got final best features {msg}")
                # Generate the best features list
                # best_features = ['x', 'y', 'z']
                best_features = []
                best_features += msg

                # Add the last column to test_data
                # test_data.update_feature(best_features[-1])

                break

            else:
                raise Exception(f"[manager] Bad MPI tag: {msg_tag}")

        # propagation
        if should_update:
            # propagate(it, best_features, config)
            print('------------propagating')
