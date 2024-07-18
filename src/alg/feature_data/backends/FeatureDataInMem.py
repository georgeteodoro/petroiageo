import numpy as np
from abc import ABC, abstractmethod
import h5py

from feature_data.backends.FeatureDataBase import FeatureDataBase
import common


class FeatureDataInMem(FeatureDataBase):
    '''
    Implementation for in-memory feature with hdf5. Although the input file
    is hdf5, it is pre-fetched in its entirety at construction. Afterwards,
    all data is in-memory (np.ndarray) and no more storage I/O is performed.
    '''
    def __init__(self, feature_path, mpi_local_comm):
        super(FeatureDataInMem, self).__init__()
        
        # Load feature File
        feature_file_name = feature_path[feature_path.rfind('/') + 1:]
        feature_name = feature_file_name[:feature_file_name.find('.')]
        if mpi_local_comm is not None:
            # Arguments to open the parallel accessible h5 file on the correct
            # communicator (there is one per node).
            mpi_kwargs = {
                'driver': 'mpio',
                'comm': mpi_local_comm,
            }
        else:
            # This is only used for testing
            print(f"[FeatureDataInMem] WARNING: initializing FeatureDataH5 "
                  f"{feature_name} without mpio. Ignore if unittesting.")
            mpi_kwargs = {}

        feature_data = self._open_feature_file(feature_path, mpi_kwargs)

        # Pre-fetch all data
        self._feature = feature_data[:]

        self._close_feature_file()

    def filter_coords(self, coords):
        '''
        Filter points, one by one. No performance guarantee was made with this
        simple implementation.
        Do we need a generator???
        '''

        # Allocate output array
        points = np.empty((len(coords), ), np.float64)

        for (i, c) in enumerate(coords):
            points[i] = self._feature[tuple(c)]

        return points
