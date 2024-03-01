from feature_data.FeatureDatasetBase import FeatureDatasetBase
from feature_data.backends.FeatureDataInMem import FeatureDataInMem


class FeatureDatasetInMemAll(FeatureDatasetBase):
    '''
    Full in-mem implementation. All FeatureData objects are in-mem
    '''
    def __init__(self, config):
        super(FeatureDatasetInMemAll, self).__init__(config)

        # Load config
        mpi_local_comm = config.get_param("mpi_local_comm")

        # Get a list of backend references
        self._features = dict()
        for feature, feature_path in self._all_features_path_dict.items():
            self._features[feature] = FeatureDataInMem(feature_path,
                                                    mpi_local_comm)

    def get_feature(self, feature):
        '''
        Returns a backend feature object.
        '''

        return self._features[feature]
