import h5py
from typing import Dict

from inverted_learning_interface import AbstractSeismicDataLoader
import config_parser


class H5SeismicDataLoader(AbstractSeismicDataLoader):
    """
    Seismic features data loader for HDF5 files.
    All features must be .h5 files, with the same dimension and same chunking.
    If running on MPI, the MPI communicator should be local to all processes
    within the same node. Local files between distributed nodes are (obviously)
    different, and will break the file opening with a synchronization error.

    No compute intensive tasks are done outside 'load()', allowing better 
    performance profiling.

    Only the File objects are stored internally, for later closure.
    All File objects are closed on '__del__', thus encapsulating anything 
    h5-related to this class.
    """

    def __init__(self, config: config_parser.Config):
        self._config = config
        self._features_files_dict_h5 = {}

        # Compatibility flags:
        super().__init__()
        self._using_h5 = True

    def load(self) -> Dict[str, h5py.Dataset]:
        """
        Only '_config.num_features' features are opened and returned.
        """

        # Select features to open
        num_features = self._config.get_param('num_features')
        features_filenames = self._config.features_files_paths
        features_names = self._config.features_files_names
        if num_features != 0:
            features_filenames = features_filenames[:num_features]
            features_names = features_names[:num_features]

        # Setup HDF5 driver configuration
        if self._config.get_param('mpi_size') > 1:
            mpi_kwargs = {
                'driver': 'mpio',
                'comm': self._config.get_param('mpi_local_comm')
            }
        else:
            mpi_kwargs = {}

        # Generate a dictionary of feature files indexed by name
        features_dset_dict_h5 = {}
        last_dim = None
        for feature_path, feature in zip(features_filenames, features_names):
            if '.h5' not in str(feature_path):
                continue
            self._features_files_dict_h5[feature] = h5py.File(
                feature_path, 'r', **mpi_kwargs)
            features_dset_dict_h5[feature] = self._features_files_dict_h5[
                feature]['f']

            # TODO: fix this assertion
            # # Assert whether the dimensions are compatible
            # if (last_dim == None) or (features_dset_dict_h5[feature].shape
            #                           == last_dim):
            #     last_dim = features_dset_dict_h5[feature].shape
            # else:
            #     raise Exception

        return features_dset_dict_h5

    def _single_compatible(self, to_compare):
        # Check if to_compare have h5 support
        compatible = to_compare._using_h5

        return compatible

    def __del__(self):
        for file in self._features_files_dict_h5.keys():
            self._features_files_dict_h5[file].close()
