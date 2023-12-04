import numpy as np
from abc import ABC, abstractmethod
import h5py

from datasets_names import FEAT_DSET_NAME
from FeatureDataBase import FeatureDataBase


class FeatureDataH5(FeatureDataBase):
    '''
    Implementation for out-of-core feature with hdf5.
    '''
    def __init__(self, feature_path, mpi_local_comm):
        super(FeatureDataH5, self).__init__(feature_path)
        feature_file_name = feature_path[feature_path.rfind('/') + 1:]
        feature_name = feature_file_name[:feature_file_name.find('.')]

        # note to self below... delete later
        # mpi_local_comm = config.get_param("mpi_local_comm")

        if mpi_local_comm is not None:
            # Arguments to open the parallel accessible h5 file on the correct
            # communicator (there is one per node).
            mpi_kwargs = {
                'driver': 'mpio',
                'comm': mpi_local_comm,
            }
            self._feature_file = h5py.File(feature_path, "r", **mpi_kwargs)
        else:
            # This is only used for testing
            print(f"[FeatureDataH5] WARNING: initializing FeatureDataH5 "
                  f"{feature_name} without mpio. Ignore if unittesting.")
            self._feature_file = h5py.File(feature_path, "r")


        assert self._feature_file is not None, "[FeatureDataH5] "\
            f"Could not open file {feature_path}"

        self._feature = self._feature_file[FEAT_DSET_NAME]

        assert self._feature_file is not None, "[FeatureDataH5] "\
            f"Could not get dataset {FEAT_DSET_NAME} of file {feature_path}"

    def __del__(self):
        self._feature_file.close()

    def filter_coords(self, coords):
        '''
        Filter points, one by one. No performance guarantee was made with this
        simple implementation.
        '''

        # Allocate output array
        points = np.empty((len(coords), ), np.float64)

        for (i, c) in enumerate(coords):
            points[i] = self._feature[tuple(c)]

        return points