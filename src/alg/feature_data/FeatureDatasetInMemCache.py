from multiprocessing import shared_memory
import fasteners  # inter-process, intra-node lock
from mpi4py import MPI
from time import monotonic_ns
from asyncio import Future
import asyncio
import numpy as np
from math import prod

from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataInShm import FeatureDataInShm
from feature_data.backends.FeatureDataH5 import FeatureDataH5


class FeatureDatasetInMemCache(FeatureDatasetBase):
    '''
    Feature caching implementation. Each feature is cached entirely.
    No pre-fetching is done. The first access is always a cache miss.
    There is a configuration for cache size: number of features loaded.

    The cache is a set of shared-memory spaces with the size of a full 
    feature. This is accessed through FeatureDataInShm objects, which can
    access this data though a numpy ndarray interface.

    The cache is LRU, managed my two lists and a set of file locks. The
    first list holds the name of the feature in a given cache line. The
    second list holds a monotonic time value to find the LRU element.
    Each feature have a read-write lock, with the write lock being used when
    the feature is loaded into the cache. No feature with a read lock can be 
    evicted from cache. On a cache miss with all cache lines filled and busy 
    (i.e., with read locks), the missing process will keep re-attempting to
    evict a cache line until one is available.

    The LRU algorithm is entirely in a critical region, protected by a global 
    lock. The cache is for processes within a common node, thus the LRU lock
    is also shared between processes on the same node. The LRU lock does not
    lingers on a process on cache misses, of cache hits with a write-locked
    feature. Instead, the algorithm releases the lock and re-attempt the 
    read process.
    '''
    def __init__(self, config):
        super(FeatureDatasetInMemCache, self).__init__(config)

        # Load config
        self._mpi_local_comm = config.get_param('mpi_local_comm')
        mpi_rank = config.get_param('mpi_rank')
        self._feature_shape = config.get_param('feature_shape')
        self._max_cache_lines = int(config.alg['feature_cache_lines'])

        # The LRU list is shared across all FeatureDatasetInMemCache within the
        # same node. LRU resolution needs to be thread-safe.
        # First, create a inter-process, intra-node file lock
        self._lru_lock = fasteners.InterProcessLock(
            '/tmp/FeatureDatasetInMemCache.lock')

        # Then, set a single intra-node process as the creator of the
        # shared-memory LRU list. This is the responsible rank.
        rank_list = np.zeros(self._mpi_local_comm.Get_size(), dtype=np.int64)
        self._mpi_local_comm.Allgather([np.int64(mpi_rank), MPI.LONG],
                                       [rank_list, MPI.LONG])
        self._is_resp_rank = mpi_rank == rank_list.min()

        # Process with the smallest rank value creates the LRU shared-data list
        if self._is_resp_rank:
            # Allocate shared-memory space
            self._shm_lru = shared_memory.SharedMemory(
                create=True,
                size=(2 * self._max_cache_lines * np.dtype('int32').itemsize))

            # Send the shm region name to all other processes
            self._mpi_local_comm.bcast(self._shm_lru.name, root=mpi_rank)
        else:
            # Get the name of the shared-memory space
            lru_shm_name = self._mpi_local_comm.bcast(None,
                                                      root=rank_list.min())

            # Remaining processes access existing LRU shared-memory
            self._shm_lru = shared_memory.SharedMemory(name=lru_shm_name,
                                                       create=False)

        # Create a numpy array which reads from the shared-memory
        self._lru = np.ndarray((2, self._max_cache_lines),
                               dtype=np.int32,
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
                self._shm_cache_lines.append(
                    shared_memory.SharedMemory(name=self._shm_feature_name(i),
                                               create=True,
                                               size=feature_size))
        self._mpi_local_comm.Barrier()

        # Remaining processes initialize shared-memory object
        if not self._is_resp_rank:
            for i in range(self._max_cache_lines):
                self._shm_cache_lines.append(
                    shared_memory.SharedMemory(name=lru_shm_name,
                                               create=False))

        # Create a list of features locks for checking if a feature can be
        # evicted from cache
        self._feature_locks = []
        for i in range(self._max_cache_lines):
            self._feature_locks.append(
                fasteners.InterProcessReaderWriterLock(self._shm_lock_path(i)))

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
        with self._lru_lock:
            # Check for cache miss
            if feature_idx not in self._lru[self._LRU_F_IDX]:
                # Get the sorted indices of cache lines, ordered by _LRU_TIME
                preference_list = np.argsort(self._lru[self._LRU_TIME])

                # Attempt to find a free cache-line
                for i in preference_list:
                    found = self._feature_locks[i].acquire_write_lock(
                        blocking=False)
                    if found:
                        line_idx = i
                        break

                # If all cache lines are in use, return and try again
                if not found:
                    return None

                # The i-th entry can be evicted.
                # Update the cache register
                self._lru[self._LRU_F_IDX][line_idx] = feature_idx

                # Set a feature path in order to load it from disk to memory
                feature_path = self._all_features_path_dict[feature]

            else:
                # Retrieve index of cache line for the hit feature
                line_idx = np.where(
                    self._lru[self._LRU_F_IDX] == feature_idx)[0][0]

                # An empty feature path represents a cache hit, i.e., no need
                # for reloading the feature into memory
                feature_path = None

            # Set a cache hit
            self._lru[self._LRU_TIME][line_idx] = monotonic_ns()

            # Create the shared-memory FeatureData wrapper asynchronously
            asyncio.create_task(
                _create_FeatureData(create_future,
                                    self._shm_feature_name(line_idx),
                                    self._shm_lock_path(feature_idx),
                                    self._feature_shape, feature_path))

        feature = await create_future
        return feature

    def get_feature(self, feature):
        '''
        Runs the async version of get_feature.
        '''

        ret = None
        # ret can be None on a cache miss which could not find a free cache line
        # to evict. On this case, it should keep trying.
        while ret is None:
            ret = asyncio.run(self._async_get_feature(feature))
        return ret

    def _shm_feature_name(self, f_id):
        # The header of self._shm_lru.name ensures that a shared-memory space
        # is always available, even if the program had previously exited
        # with errors
        return f'{self._shm_lru.name}FeatureDatasetInMemCache.feature{f_id}'

    def _shm_lock_path(self, f_id):
        return f'/tmp/FeatureDatasetInMemCache.CacheLine{f_id}'

    def _lru_hit(self, feature):
        '''
        For the LRU list, the first element (0) is the LRU element. On a 
        cache hit, a feature name is removed from the list and appended 
        last (_max_cache_lines-1).
        '''

        # Hit check is thread-safe across all processes within the node
        with self._lru_lock:
            if feature in self._lru:
                self._lru.pop(self._lru.index(feature))
            self._lru.append(feature)
