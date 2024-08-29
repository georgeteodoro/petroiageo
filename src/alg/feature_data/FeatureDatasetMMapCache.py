from multiprocessing import shared_memory
import fasteners  # inter-process, intra-node lock
import posix_ipc
from mpi4py import MPI
from time import monotonic_ns
import numpy as np
from functools import partial

from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataMMap import FeatureDataMMap


class FeatureDatasetMMapCache(FeatureDatasetBase):
    '''
    Feature caching implementation. Each feature is cached entirely.
    No pre-fetching is done at construction. The first access is always 
    a cache miss. There is a configuration for cache size: maximum 
    number of features loaded.

    Since features are binary npy files, there is no need to load them 
    explicitly into memory. Features are then memory-mapped instead of 
    copied into shared memory. Feature data is loaded implicitly through 
    page cache misses. Thus, this class just keeps track of all features 
    which should be in memory, respecting the limit of cache lines. 

    The cache is LRU, managed my three lists and a global lock. The first
    list holds the ID of a feature in a given cache line. The second list 
    holds a monotonic time value to find the LRU element. The third list holds
    a counter of how many processes are currently using the cache line.
    Since files are mem-mapped, access to cache pages (being them missed 
    or already in memory) are always thread-safe (the kernel guarantees it).
    On a cache miss in get_feature() with no free cache lines, the caller 
    process is blocked on a spinlock until an unused line can be evicted. 

    The LRU algorithm is entirely in a critical region, protected by a global 
    lock. The cache is for processes within a common node, thus the LRU lock
    is also shared between processes on the same node. The LRU lock does not
    lingers on a process on cache misses. If there are free cache lines, the
    algorithm releases the lock before actually loading the feature data, but
    after accounting for this new feature.
    '''

    def __init__(self, config):
        super(FeatureDatasetMMapCache, self).__init__(config)

        # Load config
        self._mpi_local_comm = config.get_param('mpi_local_comm')
        mpi_local_rank = self._mpi_local_comm.Get_rank()
        max_cache_lines = int(config.alg['feature_cache_lines'])
        self._disp_window = config.wells['window']

        # The LRU list is shared across all FeatureDatasetInMemCache within the
        # same node. LRU resolution needs to be thread-safe.
        # First, create a inter-process, intra-node file lock
        self._lru_lock = fasteners.InterProcessLock(
            '/tmp/FeatureDatasetInMemCache.lock')

        # Then, set a single intra-node process as the creator of the
        # shared-memory LRU list. This is the responsible rank.
        rank_list = np.zeros(self._mpi_local_comm.Get_size(), dtype=np.int64)
        self._mpi_local_comm.Allgather([np.int64(mpi_local_rank), MPI.LONG],
                                       [rank_list, MPI.LONG])
        self._is_resp_rank = mpi_local_rank == rank_list.min()

        # Process with the smallest rank value creates the LRU shared-data list
        if self._is_resp_rank:
            # Allocate shared-memory space
            self._shm_lru = shared_memory.SharedMemory(
                create=True,
                size=(3 * max_cache_lines * np.dtype('int64').itemsize))

            # Send the shm region name to all other processes
            self._mpi_local_comm.bcast(self._shm_lru.name, root=mpi_local_rank)
        else:
            # Get the name of the shared-memory space
            lru_shm_name = self._mpi_local_comm.bcast(None,
                                                      root=rank_list.min())

            # Remaining processes access existing LRU shared-memory
            self._shm_lru = shared_memory.SharedMemory(name=lru_shm_name,
                                                       create=False)

        # Create a numpy array which reads from the shared-memory
        self._lru = np.ndarray((3, max_cache_lines),
                               dtype=np.int64,
                               buffer=self._shm_lru.buf)

        # Setup LRU list. It is a combination of three lists. The first row [0]
        # contains the ID of the feature currently loaded on cache. The second
        # list [1] contains a list of times of when an element was last used.
        # The element with the lowest value is the LRU element. All elements
        # are initialized with -1, representing an empty element. The third and
        # final list is a counter of how many processes are reading the current
        # feature.
        self._LRU_F_IDX = 0
        self._LRU_TIME = 1
        self._LRU_COUNT = 2
        if self._is_resp_rank:
            self._lru[:, :] = -1
            # Set 0 as the count of processes using the feature
            self._lru[self._LRU_COUNT, :] = 0
        self._mpi_local_comm.Barrier()

        # Create a semaphore to act as a lockable counter of how many
        # free cache lines are available.
        # Since a semaphore is unique by node, only a single semaphore
        # per individual node is created, with all processes within it
        # having the same semaphore.
        semaphore_name = '/FeatureDatasetMMapCache.sem'
        try:
            posix_ipc.unlink_semaphore(semaphore_name)
            pass
        except Exception as e:
            print(e)
        if self._is_resp_rank:
            self._free_cache_lines_sem = posix_ipc.Semaphore(
                semaphore_name,
                flags=posix_ipc.O_CREX,
                initial_value=max_cache_lines)
            self._mpi_local_comm.Barrier()
        else:
            self._mpi_local_comm.Barrier()
            self._free_cache_lines_sem = posix_ipc.Semaphore(semaphore_name,
                                                             flags=0)

    def __del__(self):

        self._shm_lru.close()
        self._mpi_local_comm.Barrier()

        if self._is_resp_rank:
            self._shm_lru.unlink()
            self._free_cache_lines_sem.unlink()
        self._mpi_local_comm.Barrier()

        self._free_cache_lines_sem.close()

    def get_feature(self, feature):
        '''
        Runs the async version of get_feature.
        '''

        feature_idx = list(self._all_features_path_dict.keys()).index(feature)
        feature_path = self._all_features_path_dict[feature]
        done_feature_callback = partial(self.done_reading_callback, feature)

        # Flag for debugging: avoid multiple prints of 'no free cache line'
        no_free_cache_print = False
        debug = False

        # Keep trying until a feature is returned
        while True:
            self._lru_lock.acquire()
            if feature_idx in self._lru[self._LRU_F_IDX]:
                no_free_cache_print = False

                # Cache hit of feature_idx
                # Retrieve the cache line index of the hit feature
                line_idx = np.where(
                    self._lru[self._LRU_F_IDX] == feature_idx)[0][0]

                if debug:
                    print(f"[FeatureDatasetMMapCache][get_feature] hit line "
                          f"{line_idx} of feature {feature_idx}")

                # It is possible to hit a feature which has no current
                # readers. In this case the sem.release() was already called.
                # Thus, by incrementing the in-use-count of 0, we must also
                # decrement the free_cache_lines_sem counter
                if self._lru[self._LRU_COUNT][line_idx] == 0:
                    self._free_cache_lines_sem.acquire(0)

                # Update LRU value and increment in-use counter
                self._lru[self._LRU_COUNT][line_idx] += 1
                self._lru[self._LRU_TIME][line_idx] = monotonic_ns()

                # Done. Release lock and return the feature object
                self._lru_lock.release()
                return FeatureDataMMap(feature_path,
                                       done_feature_callback,
                                       disp_window=self._disp_window,
                                       pre_fetch=True)
            else:
                # Cache miss
                if not no_free_cache_print:
                    print(f"[FeatureDatasetMMapCache][get_feature] miss on "
                          f"feature {feature_idx} "
                          f"sem: {self._free_cache_lines_sem.value}")

                try:
                    # Perform non-blocking acquire
                    self._free_cache_lines_sem.acquire(0)

                    no_free_cache_print = False

                    # If passed, there are free cache lines. Count of free
                    # cache lines were decremented as a result

                    # Get the sorted indices of cache lines, ordered by _LRU_TIME
                    preference_list = np.argsort(self._lru[self._LRU_TIME])

                    # Find best cache line to "evict"
                    for i in preference_list:
                        # print(f"[FeatureDatasetMMapCache][get_feature] "
                        #       f"line {i} count "
                        #       f"{self._lru[self._LRU_COUNT][i]}")
                        if self._lru[self._LRU_COUNT][i] == 0:
                            # Found line i
                            line_idx = i
                            break

                    if debug:
                        print(f"[FeatureDatasetMMapCache][get_feature] replacing "
                              f"line {line_idx}")

                    # Update "evicted" cache line
                    self._lru[self._LRU_F_IDX][line_idx] = feature_idx
                    self._lru[self._LRU_COUNT][line_idx] += 1
                    self._lru[self._LRU_TIME][line_idx] = monotonic_ns()

                    self._lru_lock.release()
                    return FeatureDataMMap(feature_path,
                                           done_feature_callback,
                                           disp_window=self._disp_window,
                                           pre_fetch=True)

                except posix_ipc.BusyError:
                    # There are no free cache lines, thus release lock and
                    # try again...
                    if not no_free_cache_print:
                        no_free_cache_print = True
                        if debug:
                            print(f"[FeatureDatasetMMapCache][get_feature] "
                                  f"no free lines")
                    self._lru_lock.release()

    def done_reading_callback(self, feature):
        '''
        Signals that a caller process is done with a feature.
        If it is the last accessing process, then free the cache line
        '''

        feature_idx = list(self._all_features_path_dict.keys()).index(feature)
        line_idx = np.where(self._lru[self._LRU_F_IDX] == feature_idx)[0][0]

        # print(f"[FeatureDatasetMMapCache][done_reading_callback] done with "
        #       f"feature {feature_idx} on line {line_idx} with count "
        #       f"{self._lru[self._LRU_COUNT][line_idx]}")

        with self._lru_lock:
            # Decrement use counter
            self._lru[self._LRU_COUNT][line_idx] -= 1

            # Release 1 cache line if it was the last process
            if self._lru[self._LRU_COUNT][line_idx] == 0:
                self._free_cache_lines_sem.release()
