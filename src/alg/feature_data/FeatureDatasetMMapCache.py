from multiprocessing import shared_memory
import fasteners  # inter-process, intra-node lock
from mpi4py import MPI
from time import monotonic_ns, sleep
from asyncio import Future
import asyncio
import numpy as np
from math import prod

from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataInShm import FeatureDataInShm
from feature_data.backends.FeatureDataH5 import FeatureDataH5


class FeatureDatasetMMapCache(FeatureDatasetBase):
    '''
    Feature caching implementation. Each feature is cached entirely.
    No pre-fetching is done at construction. The first access is always 
    a cache miss. There is a configuration for cache size: maximum 
    number of features loaded.

    Since features are binary npy files, there is no need to load them.
    Features are then memory-mapped instead of explicitly copied into 
    shared memory.

    This class just keeps track of all features which should be in memory,
    respecting the limit of cache lines. 

    The cache is LRU, managed my two lists and a set of file locks. The
    first list holds the name of the feature in a given cache line. The
    second list holds a monotonic time value to find the LRU element.
    Since files are mem-mapped, access to cache pages (being them missed 
    or already in memory) are always thread-safe (the kernel guarantees it).
    On a cache miss in get_feature() with no free cache lines, the caller 
    process is blocked until an unused line can be evicted. 

    TODO: change implementation to semaphore!!!

    The LRU algorithm is entirely in a critical region, protected by a global 
    lock. The cache is for processes within a common node, thus the LRU lock
    is also shared between processes on the same node. The LRU lock does not
    lingers on a process on cache misses. Instead, the algorithm releases 
    the lock and re-attempt the read process.




    NEW IMPL:
    free_cache_lines = semaphore(max_cache_lines)
    cache_lines = [(feature, last_time_access, in_use_count)]

    get_feature(f):
        while True:
            self.lock()
            if f in cache_lines:
                # cache hit
                cache_lines[f].in_use_count ++
                cache_lines[f].last_time_access = time()
                self.unlock()
                return FeatureDataMMap(f)
            else:
                if free_cache_lines > 0:
                    # cache miss, being possible to load feature
                    free_cache_lines.wait()
                    cache_lines[f].in_use_count ++
                    cache_lines[f].last_time_access = time()
                    self.unlock()
                    return FeatureDataMMap(f)
                else:
                    # cache miss, no free line
                    self.unlock()
                    free_cache_lines.wait()

    done_reading_callback(f):
        self.lock()
        cache_lines[f].in_use_count ++
        if cache_lines[f].in_use_count == 0:
            free_cache_lines.signal()
        self.unlock()














    '''




    def __init__(self, config):
        super(FeatureDatasetMMapCache, self).__init__(config)

        # Load config
        self._mpi_local_comm = config.get_param('mpi_local_comm')
        mpi_local_rank = self._mpi_local_comm.Get_rank()
        self._feature_shape = config.get_param('feature_shape')
        self._max_cache_lines = int(config.alg['feature_cache_lines'])

        # The LRU list is shared across all FeatureDatasetMMapCache within the
        # same node. LRU resolution needs to be thread-safe.
        # First, create a inter-process, intra-node file lock
        self._lru_lock = fasteners.InterProcessLock(
            '/tmp/FeatureDatasetMMapCache.lock')

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
                size=(2 * self._max_cache_lines * np.dtype('int64').itemsize))

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
        self._lru = np.ndarray((2, self._max_cache_lines),
                               dtype=np.int64,
                               buffer=self._shm_lru.buf)

        # Setup LRU list. It is a combination of two lists. The first row [0]
        # contains the ID of the feature currently loaded on cache. The second
        # list [1] contains a list of times of when an element was last used.
        # The element with the lowest value is the LRU element. All elements
        # are initialized with -1, representing an empty element.
        self._LRU_F_IDX = 0
        self._LRU_TIME = 1
        if self._is_resp_rank:
            self._lru[:, :] = -1
        self._mpi_local_comm.Barrier()

        # Allocate the shared-memory for all features
        self._shm_cache_lines = []
        if self._is_resp_rank:

            # Create one shared-memory region per cache line
            feature_size = prod(
                self._feature_shape) * np.dtype('float64').itemsize
            for i in range(self._max_cache_lines):
                shm = shared_memory.SharedMemory(
                    name=self._shm_feature_name(i),
                    create=True,
                    size=feature_size)
                self._shm_cache_lines.append(shm)

                # Force allocation of shm region
                shm_array = np.ndarray(self._feature_shape,
                                       dtype=np.float64,
                                       buffer=shm.buf)
                shm_array.fill(0)

            print(f"[FeatureDatasetMMapCache][__init__] "
                  f"resp_rank allocated all shm lines.")

        self._mpi_local_comm.Barrier()

        # Remaining processes initialize shared-memory object
        if not self._is_resp_rank:
            for i in range(self._max_cache_lines):
                self._shm_cache_lines.append(
                    shared_memory.SharedMemory(name=self._shm_feature_name(i),
                                               create=False))

        # Create a list of features locks for checking if a feature can be
        # evicted from cache
        # self._feature_locks = []
        # for i in range(self._max_cache_lines):
        #     self._feature_locks.append(
        #         fasteners.InterProcessReaderWriterLock(self._shm_lock_path(i)))

    def __del__(self):

        for shm in self._shm_cache_lines:
            shm.close()
        self._shm_lru.close()
        self._mpi_local_comm.Barrier()

        if self._is_resp_rank:
            self._shm_lru.unlink()
            for shm in self._shm_cache_lines:
                shm.unlink()
        self._mpi_local_comm.Barrier()

    async def _async_get_feature(self, feature):
        '''
        Returns a shared-memory backend feature object.
        An LRU lock is used to ensure thread-safeness for LRU resolution, 
        acquired in the beginning of this routine before checking for a 
        cache hit.

        On cache miss, an attempt to load the missed feature is done. If all
        cache lines have readers, the LRU lock is released and the process 
        is re-attempted from the beginning. If there is at least one cache line
        free, this line is evicted, a write lock is set for the new feature, 
        The LRU lock is released and only then the feature data is loaded. This
        avoids prolonged periods on which the LRU lock is maintained by a 
        feature-loading process. 

        On cache hit, a read lock is acquired to the hit feature. This is a 
        blocking operation, with the process left waiting if a feature is 
        being loaded into memory by a process with a read lock. If no write
        lock exist for the hit feature, it is promptly returned. Only after
        the read lock operation is attempted, the LRU lock is released. To
        avoid waiting the end of a write lock, the read lock acquisition is
        done through futures.
        '''

        # print(f"[FeatureDatasetMMapCache][_async_get_feature] Entering")

        # Find an index value for the current feature
        feature_idx = list(self._all_features_path_dict.keys()).index(feature)

        # Future object for creating and returning the FeatureData wrapper.
        # This async execution allows the lru lock to be released without
        # waiting the creation process, which can take a long time and does
        # not requires thread-safeness.
        create_future = Future()

        # Create an async task for creating the FeatureData wrapper
        async def _create_FeatureData(f, shm_path, lock_path, feature_shape,
                                      feature_path):
            f.set_result(
                FeatureDataInShm(shm_path, lock_path, feature_shape,
                                 feature_path))

        # All LRU operations are serialized
        # print(f"[FeatureDatasetMMapCache][_async_get_feature] Getting lock")
        with self._lru_lock:
            # print(f"[FeatureDatasetMMapCache][_async_get_feature] lru_locker")

            # Check for cache miss
            if feature_idx not in self._lru[self._LRU_F_IDX]:
                print(f"[FeatureDatasetMMapCache][_async_get_feature] "
                      f"feature {feature_idx}:{feature} cache_miss")

                # Get the sorted indices of cache lines, ordered by _LRU_TIME
                preference_list = np.argsort(self._lru[self._LRU_TIME])

                # Attempt to find a free cache-line
                for i in preference_list:
                    # Perform a try-lock
                    # print(f"[FeatureDatasetMMapCache][_async_get_feature] "
                    #       f"Checking free cache line {i} with try-lock "
                    #       f"{self._feature_locks[i].path}")
                    # found = self._feature_locks[i].acquire_write_lock(
                    #     blocking=False)
                    found = True
                    if found:
                        # Release the try-lock since the FeatureData
                        # also acquires a write lock on creation
                        # print(f"[FeatureDatasetMMapCache][_async_get_feature]"
                        #       f" Found free line, releasing write-lock")
                        
                        # self._feature_locks[i].release_write_lock()
                        line_idx = i
                        break
                    print(f"[FeatureDatasetMMapCache][_async_get_feature] "
                          f"Cache line {i} in use")

                # If all cache lines are in use, return and try again
                if not found:
                    print(f"[FeatureDatasetMMapCache][_async_get_feature] "
                          f"No empty cache line")
                    return None

                print(f"[FeatureDatasetMMapCache][_async_get_feature] "
                      f"Writing feature {feature_idx}:{feature} on line "
                      f"{line_idx} of prev feature "
                      f"{self._lru[self._LRU_F_IDX][line_idx]}")

                # The i-th entry can be evicted.
                # Update the cache register
                self._lru[self._LRU_F_IDX][line_idx] = feature_idx

                # Set a feature path in order to load it from disk to memory
                feature_path = self._all_features_path_dict[feature]

            else:
                # Retrieve index of cache line for the hit feature
                line_idx = np.where(
                    self._lru[self._LRU_F_IDX] == feature_idx)[0][0]

                print(f"[FeatureDatasetMMapCache][_async_get_feature] "
                      f"Cache hit of feature {feature_idx}:{feature} on "
                      f"line {line_idx}")

                # An empty feature path represents a cache hit, i.e., no need
                # for reloading the feature into memory
                feature_path = None

            # Set a cache hit
            self._lru[self._LRU_TIME][line_idx] = monotonic_ns()

            # Create the shared-memory FeatureData wrapper asynchronously
            asyncio.create_task(
                _create_FeatureData(create_future,
                                    self._shm_feature_name(line_idx),
                                    # self._shm_lock_path(line_idx),
                                    self._mpi_local_comm,
                                    self._feature_shape, feature_path))

        feature = await create_future
        print(f"[FeatureDatasetMMapCache][_async_get_feature] "
              f"Feature {feature_idx}:{feature} is ready. Returned.")
        return feature

    def get_feature(self, feature):
        '''
        Runs the async version of get_feature.
        '''

        ret = None
        # ret can be None on a cache miss which could not find a free cache
        # line to evict. On this case, it should keep trying.
        while ret is None:
            ret = asyncio.run(self._async_get_feature(feature))
            if ret is None:
                print(f"[FeatureDatasetMMapCache][get_feature] Failed "
                      f"to get file, trying again")
                # Retry in 1 sec
                sleep(1)
        return ret

    def _shm_feature_name(self, f_id):
        # The header of self._shm_lru.name ensures that a shared-memory space
        # is always available, even if the program had previously exited
        # with errors
        return f'{self._shm_lru.name}FeatureDatasetMMapCache.feature{f_id}'

    def _shm_lock_path(self, f_id):
        return f'FeatureDatasetMMapCache.CacheLine{f_id}.lock'
