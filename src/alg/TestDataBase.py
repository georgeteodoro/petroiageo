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
        # Set the datatype for points
        self._n_features = n_features
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
        self._cur_data_type += [(f'f{f}', np.float64)
                                for f in range(n_features)]
        self._cur_data_type = np.dtype(self._cur_data_type)

        # Attributes for internal chunking, if necessary
        self._n_training_chunks = -1
        self._chunk_size = -1

        # Internal state
        self._current_feature_id = -1
        self._current_it = -1
        self._current_features = []

        # Location of concrete test_data. This should be initialize, accessed
        # and read through concrete backend subclass hooks. On the current
        # implementation, test_data is a map of points per ring. Thus it is
        # easier to sample, reshape and add more points.
        self._test_data_dict = dict()

        # Current actual size of each ring. This size increases due to
        # out-of-core porosity_data, which is ran through one chunk at a time
        self._test_data_size = dict()

        # Number of points within test_data
        self._data_len = -1

        self._wells_list = list(enumerate(wells_list))

    def _append_ring(ring, data):
        '''
        Add data to a test_data ring, updating internally its size
        '''
        size = len(data)
        beg = self._test_data_size[ring]
        end = beg + size
        self._test_data_size[ring] += size

        if self._features_only:
            self._test_data_dict[ring][
                'x',
                'y',
                'z',
                'phi',
                beg:end,
            ] = data[['x', 'y', 'z', 'phi']]
        else:
            self._test_data_dict[ring][
                'x',
                'y',
                'z',
                'phi',
                'well_id',
                beg:end,
            ] = data[['x', 'y', 'z', 'phi', 'well_id']]

    def prepare_poroisity(self, it):
        '''
        Allocate data required for the current iteration it. If data was 
        already allocated, then perform a reallocation.
        Fill coordinates, phi and well_id (when necessary).
        It also resets the internal current column.
        '''

        # Reset internal state
        self._current_it = it
        self._current_feature_id = 0
        self._current_features = []

        # Remove, if necessary, old data from previous rings
        # This should be done if sampling is required

        # Create new ring data
        self._test_data_dict[it] = np.empty((_ring_size(it, depth)),
                                            dtype=self._cur_data_type)
        self._test_data_size[it] = 0

        # Iterate on all porosity chunks to fill test_data
        for chunk_slice in porosity_data_h5.iter_chunks():
            # Skip this chunk if there are not any points withing it
            if not chunk_has_points:
                continue

            # Load porosity data chunk
            chunk_np = porosity_data_h5[chunk_slice]

            # Fill ring dict
            self._f_sel_filter.set_ring(it)
            filt_list = self._f_sel_filter.satisfies(chunk_np)
            self._append_ring(it, chunk_np[filt_list])

        # Update size and chunking info
        self._chunk_size = 0
        for r in self._test_data_dict.keys():
            self._chunk_size += len(self._test_data_dict[r])

    def commit_feature(self):
        '''
        Commits the current feature, then setting up the next feature.
        '''
        assert self._current_feature_id >= 0, "[TestDataBase][commit_feature] "\
            "Committing feature before prepare_poroisity."
        assert self._current_feature_id < self._n_features, \
            "[TestDataBase][commit_feature] Committing beyond last feature."

        self._current_feature_id += 1
        self._current_features.append(f'f{self._current_feature_id}')

    def update_feature(self, data):
        '''
        Adds data to the last feature. Data is related to all rings.
        '''

        f_str = f'f{self._current_feature_id}'

        # Fill data, one ring at a time
        for r in self._test_data_dict.keys():
            data_from_ring = data[data['ring'] == r]

            # THIS IS BACKEND-SPECIFIC!!!!!!
            self._test_data_dict[r][f_str] = data_from_ring

    def _get_values(self, well_filter, chunk_id):
        '''
        Helper function for filtering test_data.
        Returns the number of filtered points.
        '''

        # Fill training data, one ring at a time
        X = []
        y = []
        for r in self._test_data_dict.keys():

            # CHUNKING NOT IMPLEMENTED
            # Should be something like:
            # for each ring, return the ratio of |ring|/|test_data|
            # with the proper range
            if chunk_id > 0:
                raise Exception("[TestDataBase][get_train_values] Chunking "
                                "not implemented for incremental learning.")

            # Get all points from current ring, filtered by well_id
            new_points = self._test_data_dict[r]
            new_points = well_filter(new_points_X)

            # Split X from y
            new_points_X = self._test_data_dict[r][self._current_features]
            new_points_y = self._test_data_dict[r]['phi']

            # Add them to output arrays
            X.append(new_points_X)
            y.append(new_points_y)

        X = np.concatenate(X_val_list).reshape(-1)
        y = np.concatenate(y_val_list).reshape(-1)

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        X = np.array(X.tolist())
        y = np.array(y.tolist())

        return X, y

    def get_train_values(self, well_id, chunk_id):
        '''
        Leave-one-well-out validation function. Returns all data that
        is NOT on well_id. If well_id=-1, then all data is returned.
        Generates the training data inplace. X_train and y_train are
        generated inplace to avoid reallocation for them.
        If no out-of-core is used internally, then chunk_id=0 and all
        test_data is returned (filtered by well_id obviously).
        '''
    
        if not well_id or well_id < 0:
            # If well_id=-1 then no filtering is required
            well_filter = lambda data: data
        else:
            # Otherwise, returns all points outside well_id
            well_filter = lambda data: data[data['well_id'] != well_id]

        return self._get_values(well_filter, chunk_id)

    def get_val_values(well_id):
        well_filter = lambda data: data[data['well_id'] == well_id]
        # chunk_id=0 to return all data
        return self._get_values(well_filter, chunk_id=0)

        return None, None

    def set_num_training_chunks(self, n_training_chunks):
        self._n_training_chunks = n_training_chunks

    # def get_train_max_size(self):
    #     return self._chunk_size

    # def get_X_train_shape(self):
    #     return (self.get_train_expected_size(), self._current_feature_id)
