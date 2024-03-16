from datetime import datetime
import h5py
from mpi4py import MPI
from timeit import default_timer as timer
from time import time

from mpi_module import MPI_TAGS
from feature_sel import test_new_feature
from feature_data.FeatureDatasetSimple import FeatureDatasetSimple
from feature_data.FeatureDatasetInMemAll import FeatureDatasetInMemAll
from feature_data.FeatureDatasetInMemCache import FeatureDatasetInMemCache
from TrialDataNumpy import TrialDataNumpy
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
    is_feature_in_mem = config.get_param("is_feature_in_mem")
    is_feature_cache = config.get_param("is_feature_cache")
    num_its = config.alg['num_its']
    start_it = config.alg['it']
    train_wells_ids = config.train_wells_ids

    t0 = time()

    # Generate the dict of all features
    if is_feature_in_mem:
        all_features = FeatureDatasetInMemAll(config)
    elif is_feature_cache:
        all_features = FeatureDatasetInMemCache(config)
    else:
        all_features = FeatureDatasetSimple(config)
    t1 = time()
    print(f"{beg_str} Loaded all features in {t1-t0:.2f} secs.")

    # Load porosity data
    porosity_h5_f, porosity_h5_dset = _load_porosity(config)
    t2 = time()
    print(f"{beg_str} Loaded porosity in {t2-t1:.2f} secs.")

    # Prepare trial_data
    trial_data = TrialDataNumpy(train_wells_ids, porosity_h5_dset, config)
    t3 = time()
    print(f"{beg_str} Created local TrialData in {t3-t2:.2f} secs.")

    if rank_should_propagate:
        print(f"{beg_str} Porosity shape: {porosity_h5_dset.shape}.")

    print(f"{beg_str} Beginning iterations.")

    for it in range(start_it, num_its + start_it):
        t0 = time()
        start_time = timer()
        best_features = ['x', 'y', 'z']

        # Update test data: set trial_data size and update coordinates,
        # porosity, and other columns
        trial_data.prepare_porosity(it)
        t1 = time()
        print(f"{beg_str}[it{it}] Prepared trial_data in {t1-t0} secs.")

        # feature selection
        comm.send(None, dest=manager_rank, tag=MPI_TAGS.WORKER_FIRST_JOB.value)

        while True:
            # print(f"{beg_str}[it{it}] Waiting msg...")
            status = MPI.Status()
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()

            # Don't count the original [x,y,z] features
            f_it = len(best_features) - 3

            # Respond the received job
            if msg_tag == MPI_TAGS.MANAGER_NEW_JOB.value:
                # print(beg_str + f"[it{it}][f_it{f_it}] Got to test {msg}")

                # Got new feature to analyze
                new_features = msg
                results = []
                for (feature, disp) in new_features:
                    t10 = time()
                    trial_data.update_feature(
                        all_features.get_feature(feature), disp)
                    ret = test_new_feature(trial_data, config)

                    # None is returned upon only 1 well propagating.
                    # If so, propagation is halted.
                    if not ret:
                        print(f"{beg_str}[it{it}][f_it{f_it}] Only one"
                              f"remaining well on trial data. Aborting.")
                        comm.send(results,
                                  dest=manager_rank,
                                  tag=MPI_TAGS.WORKER_ABORT_PROP.value)

                        return

                    results.append(((feature, disp), *ret))
                    t11 = time()
                    print(f"{beg_str}[it{it}][f_it{f_it}] Trial "
                          f"{best_features + [(feature, disp)]} "
                          f"in {t11-t10:.2f}")

                # Send response back
                comm.send(results,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_JOB_RESULT.value)

            elif msg_tag == MPI_TAGS.MANAGER_SELECTED_FEATURE.value:
                # print(f"{beg_str}[it{it}][f_it{f_it}] New best feature {msg}")
                # Got the best feature for a f_it
                new_feature = msg
                (feature, disp) = new_feature
                best_features.append(new_feature)
                trial_data.update_feature(all_features.get_feature(feature),
                                          disp)
                trial_data.commit_feature()

                # Send response back requesting new job
                # print(f"{beg_str}[it{it}][f_it{f_it}] New first job")
                comm.send(None,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_FIRST_JOB.value)

            elif msg_tag == MPI_TAGS.MANAGER_BEST_FEATURES.value:
                # print(f"{beg_str}[it{it}][f_it{f_it}] Final features {msg}")
                # Generate the best features list
                # best_features = ['x', 'y', 'z']
                best_features = []
                best_features += msg

                # Add the last column to trial_data
                (last_feature, disp) = best_features[-1]
                trial_data.update_feature(
                    all_features.get_feature(last_feature), disp)

                break

            elif msg_tag == MPI_TAGS.MANAGER_ABORT_PROP.value:
                print(f"{beg_str}[it{it}][f_it{f_it}] Received abort.")

                return

            else:
                raise Exception(f"{beg_str} Bad MPI tag: {msg_tag}")

        t2 = time()
        print(f"{beg_str}[it{it}] Done feature_sel in: {t2-t1}")

        # Propagation
        # Only one rank per node actually commits data to the hdf5 file,
        # enforced by 'rank_should_propagate'.
        # Also, propagation can be disabled in order to experiment with
        # feature selection only, enforced by 'feature_sel_only'
        if rank_should_propagate and not feature_sel_only:
            print(f"{beg_str}[it{it}] Beginning propagation...")
            n_propagated_points = propagate(porosity_h5_dset, trial_data,
                                            all_features, best_features, it,
                                            config)
            print(f"{beg_str}[it{it}] Propagated "
                  f"{n_propagated_points} points.")

            end_time = timer()
            print(f"{beg_str}[it{it}] Iteration total "
                  f"time(s): {end_time-start_time}")

            t3 = time()
            print(f"{beg_str}[it{it}] Done propagate in: {t3-t2}")
        elif rank_should_propagate:
            print(f"{beg_str}[it{it}] SKIPPING PROPAGATION "
                  f"(feature selection only)")
            
            # Propagation barrier is also used by manager, regardless of f-sel
            print(f"{beg_str}[it{it}] Waiting fsel on skip")
            comm.Barrier()

            break

        # Wait for the end of propagation
        print(f"{beg_str}[it{it}] Waiting fsel")
        comm.Barrier()

    porosity_h5_f.close()
    if rank_should_propagate:
        print(beg_str + f' End Time(hh:mm:ss.ms): {datetime.now()}')
