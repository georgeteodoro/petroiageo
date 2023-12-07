import numpy as np
from abc import ABC, abstractmethod


def gen_features_list(config):
    '''
    Generates the list of available features with all possible displacements.
    '''
    num_features = config.get_param('num_features')
    base_features = config.features_files_names
    window_size = config.alg['window']

    # Ignore any other file which is not an .h5 file.
    base_features = [f for f in base_features if f != ".gitkeep"]
    assert len(base_features) > 0, "No features found."

    # Limit features list to the maximum size
    if num_features > 0:
        base_features = base_features[:num_features]

    # Expand features for all displacements
    all_features = []
    for f in base_features:
        for i in range(-window_size, window_size + 1):
            for j in range(-window_size, window_size + 1):
                for k in range(-window_size, window_size + 1):
                    all_features.append((f, (i, j, k)))

    return all_features


def load_all_features(config, feature_class):
    '''
    Load all available features into a dict with their 
    name and a FeatureDataH5.
    '''
    # Load config
    features_filenames = config.features_files_paths
    features_names = config.features_files_names
    num_features = config.get_param("num_features")
    mpi_local_comm = config.get_param("mpi_local_comm")

    # Load each seismic file
    all_features_dict = dict()
    total_features = 0
    for (feature_path, feature) in zip(features_filenames, features_names):
        if (num_features != 0) and (num_features == total_features):
            break
        if ".h5" not in str(feature_path):
            continue

        all_features_dict[feature] = feature_class(str(feature_path),
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
