from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataInMem import FeatureDataInMem


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
        self._mpi_local_comm = config.get_param("mpi_local_comm")
        self._max_cache_lines = 1

        # Setup empty cache
        self._features = dict()

        # Setup LRU list. It contains a list of utilized features. The first
        # element (0) is the LRU element. On a cache hit, a feature name is
        # removed from the list and appended last (_max_cache_lines-1).
        self._lru = []

        # # Get a list of backend references
        # self._features = dict()
        # for feature, feature_path in self._all_features_path_dict.items():
        #     self._features[feature] = FeatureDataInMem(feature_path,
        #                                                self._mpi_local_comm)

    def get_feature(self, feature):
        '''
        Returns a backend feature object.
        '''

        # Check for cache miss
        if feature not in self._features:
            # Evict cache line based on LRU if needed
            if len(self._lru) == self._max_cache_lines:
                lru_feature_name = self._lru.pop(0)
                lru_feature = self._features.pop(lru_feature_name)
                del lru_feature

            # Load the missed feature
            feature_path = self._all_features_path_dict[feature]
            self._features[feature] = FeatureDataInMem(feature_path,
                                                       self._mpi_local_comm)

        # Return cache hit
        self._lru_hit(feature)
        return self._features[feature]

    def _lru_hit(self, feature):
        '''
        For the LRU list, the first element (0) is the LRU element. On a 
        cache hit, a feature name is removed from the list and appended 
        last (_max_cache_lines-1).
        '''

        if feature in self._lru:
            self._lru.pop(self._lru.index(feature))
        self._lru.append(feature)
