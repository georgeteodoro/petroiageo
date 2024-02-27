import numpy as np
from abc import ABC, abstractmethod


def load_all_features(config):
    '''
    Load all available features into a dict with their 
    name and a FeatureDataH5.
    '''

    # Imports are here to avoid circular import
    from FeatureDataH5 import FeatureDataH5
    from FeatureDataInMem import FeatureDataInMem

    # Load config
    features_filenames = config.features_files_paths
    features_names = config.features_files_names
    num_features = config.get_param("num_features")
    mpi_local_comm = config.get_param("mpi_local_comm")
    is_feature_in_mem = config.get_param("is_feature_in_mem")

    # Load each seismic file
    all_features_dict = dict()
    total_features = 0
    for (feature_path, feature) in zip(features_filenames, features_names):
        if (num_features != 0) and (num_features == total_features):
            break
        if ".h5" not in str(feature_path):
            continue

        if is_feature_in_mem:
            all_features_dict[feature] = FeatureDataInMem(
                str(feature_path), mpi_local_comm)
        else:
            all_features_dict[feature] = FeatureDataH5(str(feature_path),
                                                       mpi_local_comm)
        total_features += 1

    return all_features_dict


class FeatureDataBase(ABC):
    '''
    Abstract feature_data class. Base for different feature_data backends.
    Feature data should be just a float value for a 3D hypercube space.
    '''
    def __init__(self, feature_path):
        self._feature = None

    @abstractmethod
    def filter_coords(self, coords):
        '''
        Should return a 1D array with the feature values only, on the same
        order as the input 'coords'.
        '''
        raise Exception("[FeatureDataBase][filter_coords] "
                        "Abstract method not implemented.")
