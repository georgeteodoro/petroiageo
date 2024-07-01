import numpy as np
from abc import ABC, abstractmethod
import h5py
import fasteners  # inter-process, intra-node lock
from multiprocessing import shared_memory
from time import time

from feature_data.backends.FeatureDataBase import FeatureDataBase
import common


class FeatureDataInShm(FeatureDataBase):
    '''
    Implementation for in-memory feature with hdf5. Although the input file
    is hdf5, it is pre-fetched in its entirety at construction, if necessary. 
    Afterwards, all data is in-memory (shared memory accessed through 
    np.ndarray) and no more storage I/O is performed.

    This version uses shared-memory. On creation, if the feature_path
    is valid, then the feature data is read on the shared-memory location
    from shm_path. Otherwise, it is assumed that the data is already on 
    shm_path. Either way, an np.ndarray wraps the shared-memory data.
    This means: CALLER SHOULD SYNC CREATION!!!
    If feature_path==None then data from H5 file is copied into the shared
    memory space.

    NO ALLOCATION IS DONE HERE!!! Shared memory space should already be
    allocated. Here the numpy interface is acquired and H5 data is copied,
    if necessary.

    On feature loading, a write lock is acquired, released after the 
    feature is loaded. Before the np.ndarray creation, a read lock is 
    acquired. The read lock is released at destruction. All locking is
    blocking, thus, to avoid waiting a possible feature loading process 
    futures are recommended.
    '''
    def __init__(self,
                 shm_path,
                 lock_path,
                 feature_shape,
                 feature_path=None):
        super(FeatureDataInShm, self).__init__()

        # print(f"[FeatureDataInShm][__init__] Begun")

        # Create the np.ndarray interface to the shared-memory
        self._shm_path = shm_path
        self._shm_feature = shared_memory.SharedMemory(name=shm_path,
                                                       create=False)
        self._feature = np.ndarray(feature_shape,
                                 dtype=np.float64,
                                 buffer=self._shm_feature.buf)

        self._lock = fasteners.InterProcessReaderWriterLock(lock_path)

        if feature_path is not None:
            # print(f"[FeatureDataInShm][__init__] Should read feature data.")
            # print(f"[FeatureDataInShm][__init__] Getting Write lock...")
            self._lock.acquire_write_lock()
            # print(f"[FeatureDataInShm][__init__] Got Write lock")

            # Load H5 File. mpio driver not required here since only one 
            # process per node need can fill the in-memory cache by opening
            # and reading the file.
            feature_file_name = feature_path[feature_path.rfind('/') + 1:]
            feature_name = feature_file_name[:feature_file_name.find('.')]
            feature_file = h5py.File(feature_path, 'r')

            assert feature_file is not None, "[FeatureDataInShm] "\
                f"Could not open file {feature_path}"

            feature_dset = feature_file[common.FEAT_DSET_NAME]

            assert feature_dset is not None, "[FeatureDataInShm] "\
                f"Could not get dataset {FEAT_DSET_NAME} of file {feature_path}"

            # Pre-fetch all data
            # print(f"[FeatureDataInShm][__init__] Fetching feature data")
            t0 = time()
            self._feature[:] = feature_dset[:]
            t1 = time()
            print(f"[FeatureDataInShm][__init__] Feature {feature_name} "
                  f"loaded in {t1-t0:.4f}")
            feature_file.close()

            # print(f"[FeatureDataInShm][__init__] Releasing Write lock")
            self._lock.release_write_lock()

        # Now it can read
        # print(f"[FeatureDataInShm][__init__] Getting read lock")
        self._lock.acquire_read_lock()

    def __del__(self):
        # print(f"[FeatureDataInShm][__del__] Releasing read lock")
        self._lock.release_read_lock()
        self._shm_feature.close()
        print(f"[FeatureDataInShm][__del__] closed {self._shm_path}")
        # print(f"[FeatureDataInShm][__del__] Done")

    def filter_coords(self, coords):
        '''
        Filter points, one by one. No performance guarantee was made with this
        simple implementation.
        Do we need a generator???
        '''

        # Allocate output array
        points = np.empty((len(coords), ), np.float64)

        for (i, c) in enumerate(coords):
            points[i] = self._feature[tuple(c)]

        return points