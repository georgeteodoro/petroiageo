from abc import ABC, abstractmethod


class FeatureDatasetBase(ABC):
    '''
    Abstract class for a set of all feature_data. Only interface should be the
    return of filtered coords for a given feature. Base class also maintains
    the mapping of features and paths.

    Feature data should be just a float value for a 3D hypercube space. 
    '''
    def __init__(self, config):
        # Load config
        features_filenames = config.features_files_paths
        features_names = config.features_files_names
        num_features = config.get_param("num_features")

        # Assemble a map of feature and path
        self._all_features_path_dict = dict()
        total_features = 0
        for (feature_path, feature) in zip(features_filenames, features_names):
            if (num_features != 0) and (num_features == total_features):
                # Limit the number of features
                break
            if ".h5" not in str(feature_path):
                # Skip all non-H5 files
                continue

            self._all_features_path_dict[feature] = str(feature_path)
            total_features += 1

    @abstractmethod
    def get_feature(self, feature):
        '''
        Returns a backend feature object.
        '''
        raise Exception("[FeatureDatasetBase][get_feature] "
                        "Abstract method not implemented.")
