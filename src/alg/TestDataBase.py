import numpy as np


class TestDataBase(object):
    '''
    Abstract test_data class, which implements filtering and sampling. 
    All test_data should be an array of the following columns, on this order:
     - x, y, z (point coordinates)
     - phi (porosity)
     - well_id (only required for feature selection, not propagation)
     - features (multiple columns)
    '''
    def __init__(self, n_features, features_only, wells_list):
        self._features_only = features_only
        if features_only:
            self._cur_data_type = [
                ('x', np.int64),
                ('y', np.int64),
                ('z', np.int64),
                ('phi', np.float64),
            ]
        else:
            self._cur_data_type = [
                ('x', np.int64),
                ('y', np.int64),
                ('z', np.int64),
                ('phi', np.float64),
                ('well_id', np.int64),
            ]

        self._n_training_chunks = -1
        self._chunk_size = -1

        self._current_feature = -1
        self._n_features = n_features

        self._current_it = -1

        # Location of concrete test_data. This should be initialize, accessed
        # and read through concrete backend subclass hooks. On the current
        # implementation, test_data is a map of points per ring. Thus it is
        # easier to sample, reshape and add more points.
        self._test_data = dict()

        # Coordinate of the last point of a given ring
        self._test_data_last = dict()

        # Number of points within test_data
        self._data_len = -1

        self._wells_list = list(enumerate(wells_list))

    def prepare_poroisity(it):
        '''
        Allocate data required for the current iteration it. If data was 
        already allocated, then perform a reallocation.
        Fill coordinates, phi and well_id (when necessary).
        '''

        self._current_it = it

        # Remove, if necessary, old data from previous rings

        # Create new ring data
        self._test_data[it] = np.empty((_ring_size(it, depth)),
                                       dtype=self._cur_data_type)
        self._test_data_last[it] = 0

        # Iterate on all porosity chunks
        for chunk_slice in porosity_data_h5.iter_chunks():
            # Skip this chunk if there are not any points withing it
            if not chunk_has_points:
                continue

            # Load porosity data chunk
            chunk_np = porosity_data_h5[chunk_slice]

            # Fill ring dict
            for (x, y, z, phi, real, ring, well_id) in chunk_np:
                if self._f_sel_filter.satisfies(real):
                    if self._features_only:
                        self._test_data[ring][self._test_data_last[ring]] = (
                            x, y, z, phi)
                    else:
                        self._test_data[ring][self._test_data_last[ring]] = (
                            x, y, z, phi, well_id)

            

    def commit_feature():
        pass

    def update_feature():
        pass

    def get_train_values(well_id, chunk_id, X_train, y_train):
        pass

    def get_val_values(well_id):
        return None, None

    def set_num_training_chunks(n_training_chunks):
        self._n_training_chunks = n_training_chunks

    def get_train_expected_size():
        return self._chunk_size

    def get_train_type():
        return self._cur_data_type
