import numpy as np
from abc import ABC, abstractmethod
import h5py

import common


class FeatureDataBase(ABC):
    '''
    Abstract feature_data class. Base for different feature_data backends.
    Feature data should be just a float value for a 3D hypercube space.
    '''
    def __init__(self):
        self._feature = None
        self._ext = None
        self._feature_file = None

    @abstractmethod
    def filter_coords(self, coords):
        '''
        Should return a 1D array with the feature values only, on the same
        order as the input 'coords'.
        '''
        raise Exception("[FeatureDataBase][filter_coords] "
                        "Abstract method not implemented.")

    def _open_feature_file(self, feature_path, mpi_kwargs={}):
        '''
        Returned object uses numpy interface.
        Supports indexed access and shape.
        '''

        filename = feature_path.split('/')[-1]
        self._ext = filename.split('.')[-1]

        if self._ext == 'h5':
            self._feature_file = h5py.File(feature_path, 'r', **mpi_kwargs)

            assert self._feature_file is not None, "[FeatureDataInShm] "\
                f"Could not open file {feature_path}"

            feature_data = self._feature_file[common.FEAT_DSET_NAME]

            assert feature_data is not None, "[FeatureDataInShm] "\
                f"Could not get dataset {FEAT_DSET_NAME} of file {feature_path}"
        elif self._ext == 'npy':
            feature_data = np.load(feature_path)
            pass
        else:
            raise Exception(f"[FeatureDataBase] Unknown feature extension"
                            f"{ext} for {filename}.")

        return feature_data

    def _close_feature_file(self):
        assert self._ext is not None, f"[FeatureDataBase] Attempting to "\
            "close unopened feature file."

        if self._ext == 'h5':
            self._feature_file.close()
        elif self._ext == 'npy':
            pass

        self._ext = None
        if self._feature_file is not None:
            del self._feature_file
