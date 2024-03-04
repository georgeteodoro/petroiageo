from multiprocessing import shared_memory
import fasteners  # inter-process, intra-node lock
from mpi4py import MPI

from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataInMem import FeatureDataInMem
from feature_data.backends.FeatureDataH5 import FeatureDataH5


class FeatureDatasetInMemCache(FeatureDatasetBase):
    '''
    Feature caching implementation. Each feature is cached entirely.
    No pre-fetching is done. The first access is always a cache miss.
    There is a configuration for cache size: number of features loaded.

    TODO: For this first implementation, each process has a FeatureDataset,
    meaning that the cache is individual per process. If initial tests are ok
    a shared in-node implementation will be done.
    '''
    def __init__(self, config):
        super(FeatureDatasetInMemCache, self).__init__(config)

        # Load config
        self._mpi_local_comm = config.get_param('mpi_local_comm')
        mpi_rank = config.get_param('mpi_rank')
        self._max_cache_lines = 1

        # The LRU list is shared across all FeatureDatasetInMemCache within the
        # same node. LRU resolution needs to be thread-safe.
        # First, create a inter-process, intra-node file lock
        self._lru_lock = fasteners.InterProcessLock(
            '/tmp/FeatureDataInMem.lock')

        # Then, set a single intra-node process as the creator of the
        # shared-memory LRU list.
        rank_list = np.zeros(self._mpi_local_comm.Get_size(), dtype=np.int64)
        self._mpi_local_comm.Allgather([int64(mpi_rank), MPI.LONG],
                                       [rank_list, MPI.LONG])

        # Process with the smallest rank value creates the LRU shared-data list
        lru_shm_name = 'FeatureDatasetBase.lru'
        if mpi_rank == rank_list.min():
            # Allocate shared-memory space
            self._shm_lru = shared_memory.SharedMemory(
                name=lru_shm_name,
                create=True,
                size=(2 * self._max_cache_lines * np.dtype('int32').itemsize))

        # Wait for allocation of LRU shared-memory region
        self._mpi_local_comm.Barrier()

        # Remaining processes access existing LRU shared-memory
        if mpi_rank != rank_list.min():
            # Load shared-memory space
            self._shm_lru = shared_memory.SharedMemory(name=lru_shm_name,
                                                       create=False)

        # Create a numpy array which reads from the shared-memory
        self._lru = np.array((2, self._max_cache_lines),
                             dtype=np.int32,
                             buffer=self._shm_lru.buf)

        # Setup LRU list. It is a combination of two lists. The first row [0]
        # contains a list of times of when an element was last used.
        # The element with the lowest value is the LRU element. The second
        # list [1] contains the ID of the feature currently loaded on cache.
        # All elements are initialized with -1, representing an empty element.
        if mpi_rank == rank_list.min():
            self._lru[:, :] = -1
        self._mpi_local_comm.Barrier()

        # Allocate the shared-memory for all features
        self._shm_cache_lines = []
        if mpi_rank == rank_list.min():
            # Get a sample feature to find the dimensions
            # required for a feature
            sample_feature_path = next(
                iter(self._all_features_path_dict.values()))
            sample_feature = FeatureDataH5(sample_feature_path,
                                           self._mpi_local_comm)
            feature_size = sample_feature._feature.dtype.itemsize * \
                           sample_feature._feature.size
            del sample_feature

            # Create one shared-memory region per cache line
            for i in range(self._max_cache_lines):
                self._shm_cache_lines.append(
                    shared_memory.SharedMemory(name=self._shm_feature_name(i),
                                               create=True,
                                               size=feature_size))
        self._mpi_local_comm.Barrier()

        # Remaining processes initialize shared-memory object
        if mpi_rank != rank_list.min():
            for i in range(self._max_cache_lines):
                self._shm_cache_lines.append(
                    shared_memory.SharedMemory(name=lru_shm_name,
                                               create=False))

    def get_feature(self, feature):
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

        # Check for cache miss
        if feature not in self._features:
            # Evict cache line based on LRU if needed
            if len(self._lru) == self._max_cache_lines:
                lru_feature_name = self._lru.pop(0)
                lru_feature = self._features.pop(lru_feature_name)
                del lru_feature
                print(f"[FeatureDatasetBase] Evicting: {lru_feature_name}")

            # Load the missed feature
            print(f"[FeatureDatasetBase] Loading to cache: {lru_feature_name}")
            feature_path = self._all_features_path_dict[feature]
            self._features[feature] = FeatureDataInMem(feature_path,
                                                       self._mpi_local_comm)

        # Return cache hit
        self._lru_hit(feature)
        return self._features[feature]

    def _shm_feature_name(self, f_id):
        return f'FeatureDatasetInMemCache.{f_id}'

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
