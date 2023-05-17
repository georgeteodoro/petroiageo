import h5py
from time import time

from inverted_learning_interface import AbstractFeatureSelectionAlg
import petro_dist4_hdf5
import config_parser


class H5FeatureSelectionAlg(AbstractFeatureSelectionAlg):
    """
    Feature selection algorithm for HDF5 files and MPI.
    Although MPI support is there, it is possible to run it without 'mpirun'.
    """

    def __init__(self, config: config_parser.Config):
        self._config = config

    @staticmethod
    def _generate_seismic_features_names(window, base_features) -> list:
        """
        Generate a list of the product of base_features and possible 
        displacements.
        """
        all_features = []

        for f in base_features:
            for i in range(-window, window + 1):
                for j in range(-window, window + 1):
                    for k in range(-window, window + 1):
                        all_features.append((f, i, j, k))

        return all_features

    def feature_selection(self, features_dict_h5, porosity_data_h5, it):
        # Retrieve config parameters
        max_num_features = self._config.alg['max_num_features']
        num_features = self._config.get_param('num_features')
        base_features = self._config.features_files_names[:num_features]
        window_size = self._config.get_param('window')
        max_tested_features = self._config.get_param('max_tested_features')
        mpi_size = self._config.get_param('mpi_size')

        # Calculate remaining variables
        all_features = self._generate_seismic_features_names(
            window_size, base_features)
        window_shape = (window_size, window_size, window_size)
        displacement_cube_shape = (window_size * 2 + 1, window_size * 2 + 1,
                                   window_size * 2 + 1)

        if mpi_size == 1:
            best_features_set, best_error = petro5_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features, window_shape,
                displacement_cube_shape, it, max_num_features,
                max_tested_features, self._config)
        else:
            best_features_set, best_error = petro_dist4_hdf5.get_features_sets(
                porosity_data_h5, features_dict_h5, all_features, window_shape,
                displacement_cube_shape, it, max_num_features,
                max_tested_features, self._config)

        return best_features_set

    # TODO: implement this....
    def compatible(self, to_compare):
        return True
