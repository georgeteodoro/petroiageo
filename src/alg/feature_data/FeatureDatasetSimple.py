from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataMMap import FeatureDataMMap


class FeatureDatasetSimple(FeatureDatasetBase):
    '''
    Simple implementation of feature dataset. Nothing smart is done.
    A backend reference for each feature is maintained within it, being
    directly accessed when needed. The backend can be configurable.
    '''
    def __init__(self, config):
        super(FeatureDatasetSimple, self).__init__(config)

        # Load config
        mpi_local_comm = config.get_param("mpi_local_comm")
        is_feature_in_mem = config.get_param("is_feature_in_mem")

        # Get a list of backend references
        self._features = dict()
        for feature, feature_path in self._all_features_path_dict.items():
            self._features[feature] = FeatureDataMMap(feature_path)

    def get_feature(self, feature):
        '''
        Returns a backend feature object.
        '''

        return self._features[feature]
