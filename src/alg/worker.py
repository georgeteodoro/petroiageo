from mpi4py import MPI
import h5py

from mpi_module import MPI_TAGS
from feature_sel import test_new_feature
import FeatureDataBase
from TrialDataNumpy import TrialDataNumpy
from FeatureDataH5 import FeatureDataH5
from data_filter import WellsSingleRingDataFilter
from propagate import propagate
import common

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

beg_str = f"[worker{rank}]"


def _load_porosity(config):
    # For MPI_FILE_OPEN, used by hdf5 with mpi, all files must be opened
    # with the same access/mode: existing file with write permission
    # However, only one process should update this porosity_data_h5
    # structure.
    write_str = "r+"

    # Setup HDF5 driver configuration
    mpi_kwargs = {
        'driver': 'mpio',
        'comm': config.get_param('mpi_local_comm'),
    }

    porosity_cube_file = h5py.File(config.starting_porosity_cube_path,
                                   write_str, **mpi_kwargs)
    porosity_cube_dset = porosity_cube_file[common.POROSITY_DSET_NAME]

    return porosity_cube_file, porosity_cube_dset


def run(config):
    rank_should_propagate = config.get_param('mpi_should_update_local')
    feature_sel_only = config.get_param('feature_sel_only')
    num_its = config.alg['num_its']
    start_it = config.alg['it']
    train_wells_ids = config.train_wells_ids

    # Generate the dict of all features
    all_features_dict = FeatureDataBase.load_all_features(
        config, FeatureDataH5)

    # Load porosity data
    print(beg_str + f"Loading porosity.")
    porosity_h5_f, porosity_h5_dset = _load_porosity(config)

    # Prepare trial_data
    print(beg_str + f"Preparing trial_data.")
    f_sel_filter = WellsSingleRingDataFilter(train_wells_ids)
    trial_data = TrialDataNumpy(porosity_data=porosity_h5_dset,
                                f_sel_filter=f_sel_filter,
                                config=config)

    print(beg_str + f"Beginning iterations.")

    for it in range(start_it, num_its + start_it):
        best_features = ['x', 'y', 'z']

        # Update test data: set trial_data size and update coordinates,
        # porosity, and other columns
        print(beg_str + f"[it{it}] Preparing trial_data.")
        trial_data.prepare_porosity(it)

        # feature selection
        comm.send(None, dest=manager_rank, tag=MPI_TAGS.WORKER_FIRST_JOB.value)

        while True:
            status = MPI.Status()
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()

            # Don't count the original [x,y,z] features
            f_it = len(best_features) - 3

            # Respond the received job
            if msg_tag == MPI_TAGS.MANAGER_NEW_JOB.value:
                print(
                    beg_str +
                    f"[it{it}][f_it{f_it}] Got new feature list to test {msg}")

                # Got new feature to analyze
                new_features = msg
                results = []
                for (feature, disp) in new_features:
                    trial_data.update_feature(all_features_dict[feature], disp)
                    ret = test_new_feature(trial_data, config)

                    # None is returned upon only 1 well propagating.
                    # If so, propagation is halted.
                    if not ret:
                        print(
                            beg_str + f"[it{it}][f_it{f_it}] "
                            "Only one remaining well on trial data. Aborting.")
                        comm.send(results,
                                  dest=manager_rank,
                                  tag=MPI_TAGS.WORKER_ABORT_PROP.value)

                        return

                    results.append(((feature, disp), *ret))

                # Send response back
                comm.send(results,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_JOB_RESULT.value)

            elif msg_tag == MPI_TAGS.MANAGER_SELECTED_FEATURE.value:
                print(beg_str +
                      f"[it{it}][f_it{f_it}] Got new best feature {msg}")
                # Got the best feature for a f_it
                new_feature = msg
                (feature, disp) = new_feature
                best_features.append(new_feature)
                trial_data.update_feature(all_features_dict[feature], disp)
                trial_data.commit_feature()

                # Send response back requesting new job
                comm.send(results,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_FIRST_JOB.value)

            elif msg_tag == MPI_TAGS.MANAGER_BEST_FEATURES.value:
                print(beg_str +
                      f"[it{it}][f_it{f_it}] Got final best features {msg}")
                # Generate the best features list
                # best_features = ['x', 'y', 'z']
                best_features = []
                best_features += msg

                # Add the last column to trial_data
                (last_feature, disp) = best_features[-1]
                trial_data.update_feature(all_features_dict[last_feature],
                                          disp)

                break

            elif msg_tag == MPI_TAGS.MANAGER_ABORT_PROP.value:
                print(beg_str + f"[it{it}][f_it{f_it}] Received abort.")

                return

            else:
                raise Exception(beg_str + f" Bad MPI tag: {msg_tag}")

        # Propagation
        # Only one rank per node actually commits data to the hdf5 file,
        # enforced by 'rank_should_propagate'.
        # Also, propagation can be disabled in order to experiment with
        # feature selection only, enforced by 'feature_sel_only'
        if rank_should_propagate and not feature_sel_only:
            n_propagated_points = propagate(porosity_h5_dset, trial_data,
                                            all_features_dict, best_features,
                                            it, config)
            print(beg_str +
                  f"[it{it}] Propagated {n_propagated_points} points.")

    porosity_h5_f.close()
