import h5py

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

    def load(self):
        """
        Only '_config.num_features' features are opened and returned.
        """

        # Select features to open
        num_features = self._config.get_param('num_features')
        features_filenames = self._config.features_files_paths[:num_features]
        features_names = self._config.features_files_names[:num_features]

        # Generate a dictionary of feature files indexed by name
        features_dset_dict_h5 = {}
        last_dim = None
        for feature_path, feature in zip(features_filenames, features_names):
            self._features_files_dict_h5[feature] = h5py.File(
                feature_path,
                'r',
                driver='mpio',
                comm=self._config.get_param('mpi_local_comm'))
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

    # TODO: implement this....
    def compatible(self, to_compare):
        return True

    def __del__(self):
        for file in self._features_files_dict_h5.keys():
            self._features_files_dict_h5[file].close()
