from datetime import datetime
import h5py
from mpi4py import MPI
from timeit import default_timer as timer
from time import time
import gc

from TrialDataFast import TrialDataFast

from mpi_module import MPI_TAGS
from feature_sel import test_new_feature
from feature_data.FeatureDatasetSimple import FeatureDatasetSimple
from feature_data.FeatureDatasetInMemAll import FeatureDatasetInMemAll
from feature_data.FeatureDatasetMMapCache import FeatureDatasetMMapCache
from TrialDataNumpy import TrialDataNumpy
from TrialDataSharedNumpy import TrialDataSharedNumpy
from TrialDataH5 import TrialDataH5
from TrialDataSharedH5 import TrialDataSharedH5
from TrialDataNumpyNotHier import TrialDataNumpyNotHier
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


gc_start = None
gc_times = []


def run(config):
    rank_should_propagate = config.get_param('mpi_should_update_local')
    feature_sel_only = config.get_param('feature_sel_only')
    is_feature_in_mem = config.get_param("is_feature_in_mem")
    is_feature_cache = config.get_param("is_feature_cache")
    is_shared_trial_data = config.get_param("is_shared_trial_data")
    is_non_hier_trial_data = config.get_param("is_non_hier_trial_data")
    is_h5_trial_data = config.get_param("is_h5_trial_data")
    is_h5_shared_trial_data = config.get_param("is_h5_shared_trial_data")
    num_its = config.alg['num_its']
    start_it = config.alg['it']
    train_wells_ids = config.train_wells_ids

    # Log GC usage
    def gc_callback(phase, info):
        global gc_start
        global gc_times
        if phase == 'start':
            # this indicates the function is called before garbage collection
            gc_start = time()
        else:
            # phase have only 2 possible values: 'start' and 'stop'
            duration = time() - gc_start
            gc_start = None
            gc_times += [duration]

    gc.callbacks += [gc_callback]
    print(f"{beg_str} Starting worker.")

    t0 = time()

    # Generate the dict of all features
    if is_feature_in_mem:
        all_features = FeatureDatasetInMemAll(config)
    elif is_feature_cache:
        all_features = FeatureDatasetMMapCache(config)
    else:
        all_features = FeatureDatasetSimple(config)
    t1 = time()
    print(f"{beg_str} Loaded all features in {t1-t0:.2f} secs.")

    # Load porosity data
    porosity_h5_f, porosity_h5_dset = _load_porosity(config)
    t2 = time()
    print(f"{beg_str} Loaded porosity in {t2-t1:.2f} secs.")

    #trial_data = TrialDataFast(train_wells_ids, porosity_h5_dset, config)

    # Prepare trial_data
    if is_shared_trial_data:
        trial_data = TrialDataSharedNumpy(train_wells_ids, porosity_h5_dset,
                                          config)
    elif is_non_hier_trial_data:
        trial_data = TrialDataNumpyNotHier(train_wells_ids, porosity_h5_dset,
                                          config)
    elif is_h5_trial_data:
        trial_data = TrialDataH5(train_wells_ids, porosity_h5_dset, config)
    elif is_h5_shared_trial_data:
        trial_data = TrialDataSharedH5(train_wells_ids, porosity_h5_dset,
                                       config)
    else:
        trial_data = TrialDataNumpy(train_wells_ids, porosity_h5_dset, config)
    t3 = time()
    print(f"{beg_str} Created local TrialData in {t3-t2:.2f} secs.")

    if rank_should_propagate:
        print(f"{beg_str} Porosity shape: {porosity_h5_dset.shape}.")

    print(f"{beg_str} Beginning iterations.")

    for it in range(start_it, num_its + start_it):
        t0 = time()
        start_time = timer()
        best_features = []

        # Update test data: set trial_data size and update coordinates,
        # porosity, and other columns
        trial_data.prepare_porosity(it)
        t1 = time()
        print(f"{beg_str}[it{it}] Prepared prepare_porosity in {t1-t0} secs.")

        it_wait_job_time = 0
        it_wait_response_time = 0
        it_get_f_time = 0
        it_update_f_time = 0
        it_commit_f_time = 0
        it_training_time = 0

        # Wait all processes to finish prepare_porosity
        print(f"{beg_str}[it{it}] Waiting prepare_porosity barrier")
        comm.Barrier()

        t1 = time()

        # feature selection
        comm.send(None, dest=manager_rank, tag=MPI_TAGS.WORKER_FIRST_JOB.value)

        while True:
            t2 = time()

            # print(f"{beg_str}[it{it}] Waiting msg...")
            status = MPI.Status()
            msg = comm.recv(status=status)
            msg_tag = status.Get_tag()

            t3 = time()
            # print(f"{beg_str}[it{it}] msg_wait {t3-t2:.4f}")
            it_wait_job_time += t3 - t2

            # Don't count the original [x,y,z] features
            f_it = len(best_features)

            # Respond the received job
            if msg_tag == MPI_TAGS.MANAGER_NEW_JOB.value:
                # print(beg_str + f"[it{it}][f_it{f_it}] Got to test {msg}")

                # Got new feature to analyze
                new_features = msg
                results = []
                for (feature, disp) in new_features:
                    t4 = time()
                    # open files which are not shm
                    # opn_files = psutil.Process().open_files()
                    # opn_files = [f.path for f in opn_files if 'shm' not in f.path]
                    # print(f"{beg_str}[it{it}][f_it{f_it}] Updt-feature.")
                    # print(f"{beg_str}[it{it}][f_it{f_it}] open_files: "
                    #       f"{opn_files}.")
                    f_data = all_features.get_feature(feature)
                    t5 = time()
                    trial_data.update_feature(f_data, disp)
                    t6 = time()
                    # print(f"{beg_str}[it{it}][f_it{f_it}] Test-feature.")
                    ret = test_new_feature(trial_data, config)

                    # None is returned upon only 1 well propagating.
                    # If so, propagation is halted.
                    if not ret:
                        print(f"{beg_str}[it{it}][f_it{f_it}] Only one "
                              f"remaining well on trial data. Aborting.")
                        comm.send(results,
                                  dest=manager_rank,
                                  tag=MPI_TAGS.WORKER_ABORT_PROP.value)

                        return

                    results.append(((feature, disp), *ret))
                    t7 = time()
                    print(f"{beg_str}[it{it}][f_it{f_it}] Trial "
                          f"{best_features + [(feature, disp)]} "
                          f"rmse {ret[0]} in {t7-t4:.2f}")
                    
                    print(f"{beg_str}[it{it}][trialProf] get_feature {t5-t4}")
                    print(f"{beg_str}[it{it}][trialProf] update_feature {t6-t5}")
                    print(f"{beg_str}[it{it}][trialProf] training {t7-t6}")

                    it_get_f_time += t5 - t4
                    it_update_f_time += t6 - t5
                    it_training_time += t7 - t6

                # Send response back
                t7 = time()
                comm.send(results,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_JOB_RESULT.value)
                t8 = time()
                it_wait_response_time += t8 - t7

            elif msg_tag == MPI_TAGS.MANAGER_SELECTED_FEATURE.value:
                t9 = time()
                # print(f"{beg_str}[it{it}][f_it{f_it}] New best feature {msg}")
                # Got the best feature for a f_it
                new_feature = msg
                (feature, disp) = new_feature
                best_features.append(new_feature)
                trial_data.commit_feature(all_features.get_feature(feature),
                                          disp)
                t10 = time()
                it_commit_f_time += t10 - t9

                # Send response back requesting new job
                # print(f"{beg_str}[it{it}][f_it{f_it}] New first job")
                comm.send(None,
                          dest=manager_rank,
                          tag=MPI_TAGS.WORKER_FIRST_JOB.value)
                t11 = time()
                it_wait_response_time += t11 - t10

            elif msg_tag == MPI_TAGS.MANAGER_BEST_FEATURES.value:
                t12 = time()
                # print(f"{beg_str}[it{it}][f_it{f_it}] Final features {msg}")
                # Generate the best features list
                # best_features = ['x', 'y', 'z']
                best_features = []
                best_features += msg

                # Add the last column to trial_data
                (last_feature, disp) = best_features[-1]
                trial_data.update_feature(
                    all_features.get_feature(last_feature), disp)
                t13 = time()
                it_commit_f_time += t13 - t12

                break

            elif msg_tag == MPI_TAGS.MANAGER_ABORT_PROP.value:
                print(f"{beg_str}[it{it}][f_it{f_it}] Received abort.")

                return

            else:
                raise Exception(f"{beg_str} Bad MPI tag: {msg_tag}")

        t14 = time()
        print(f"{beg_str}[it{it}][fprof] feature_sel_total {t14-t1:.2f}")
        print(f"{beg_str}[it{it}][fprof] it_wait_job_time {it_wait_job_time:.9f}")
        print(f"{beg_str}[it{it}][fprof] it_wait_response_time "
              f"{it_wait_response_time:.9f}")
        print(f"{beg_str}[it{it}][fprof] it_get_f_time {it_get_f_time:.9f}")
        print(f"{beg_str}[it{it}][fprof] it_update_f_time {it_update_f_time:.9f}")
        print(f"{beg_str}[it{it}][fprof] it_commit_f_time {it_commit_f_time:.9f}")
        print(f"{beg_str}[it{it}][fprof] it_training_time {it_training_time:.9f}")

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

    # Print all data - delete later...
    X, y = trial_data.get_train_values(-1, -1, True)
    with open(f'training_data_it{it}.log', 'w') as f:
        for t in X:
            #print(t)
            f.write(str(tuple(t)))
            f.write('\n')
    print(beg_str + f" X: {X.shape}")
    print(beg_str + f" y: {y.shape}")

  
    porosity_h5_f.close()

    print(beg_str + f"[GC] calls: {len(gc_times)} total: {sum(gc_times):.2f}")
    if rank_should_propagate:
        print(beg_str + f' End Time(hh:mm:ss.ms): {datetime.now()}')
