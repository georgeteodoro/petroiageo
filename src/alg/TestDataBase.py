import numpy as np
from abc import ABC, abstractmethod


class TestDataBase(ABC):
    '''
    Abstract test_data class, which implements filtering and sampling. 
    All test_data should be an array of the following columns, on this order:
     - x, y, z (point coordinates)
     - phi (porosity)
     - well_id (only required for feature selection, not propagation)
     - features (multiple columns)
    '''

    def __init__(self, n_features, features_only, wells_list, porosity_data):
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
        self._porosity_data = porosity_data

    # === Interface for subclasses ============================================

    # @abstractmethod
    # def _create_new_ring_hook(self):
    #     '''
    #     Should instantiate a new empty concrete data object to hold test_data
    #     values of a single ring and return it.
    #     '''
    #     raise Exception("[TestDataBase][_create_new_ring_hook] "
    #                     "Abstract method not implemented.")

    @abstractmethod
    def _append_ring_hook(self, ring, data):
        '''
        Should add data to a test_data ring, updating internally its size.
        This method is called once per porosity chunk, for all rings.
        '''
        raise Exception("[TestDataBase][_append_ring_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _update_col_from_ring_hook(self, r, feature_data):
        '''
        Should update the last column of ring r.
        It is assumed that feature_data if already filtered for points on
        ring r. Thus, feature_data should have the correct size of the 
        internal test_data for ring r.
        '''
        raise Exception("[TestDataBase][_update_col_from_ring_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _get_ring_filtered_values_hook(self, r, well_filter):
        '''
        Should return a set of points for ring r, filtered by a well_filter.
        '''
        raise Exception("[TestDataBase][_get_ring_filtered_values_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _in_well_filter_hook(self, well_id):
        '''
        Should return a function f(data) which filters data based on
        well_id and returns only the subset of data for which 
        data['well_id'] == well_id.
        '''
        raise Exception("[TestDataBase][_in_well_filter_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _not_in_well_filter_hook(self, well_id):
        '''
        Should return a function f(data) which filters data based on
        well_id and returns only the subset of data for which 
        data['well_id'] != well_id. If well_id<0, should return the
        Whole data.
        '''
        raise Exception("[TestDataBase][_not_in_well_filter_hook] "
                        "Abstract method not implemented.")

    # =========================================================================

    def prepare_porosity(self, it):
        '''
        Allocate data required for the current iteration it.
        Fill coordinates, phi and well_id (when necessary).
        It also resets the internal current column.
        '''

        # Reset internal state
        self._current_it = it
        self._current_feature_id = 0
        self._current_features = []

        # Remove, if necessary, old data from previous rings
        # This should be done if sampling is required
        pass

        # Create new ring data
        self._test_data_dict[it] = []
        self._test_data_size[it] = 0

        # Iterate on all porosity chunks to fill test_data
        for chunk_slice in self._porosity_data.iter_chunks():
            # Skip this chunk if there are not any points withing it
            if not chunk_has_points:
                continue

            # Load porosity data chunk
            chunk_np = self._porosity_data[chunk_slice]

            # Fill ring dict
            self._f_sel_filter.set_ring(it)
            self._append_ring_hook(it, chunk_np)

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

    def update_feature(self, feature_data):
        '''
        Adds data to the last feature. Data is related to all rings.
        '''

        # Fill data, one ring at a time
        for r in self._test_data_dict.keys():
            filtered_feature_data = ...  # filter feature_data by coordinates of _test_data_dict[r] coordinates
            self._update_col_from_ring_hook(r, filtered_feature_data)

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

            new_points = self._get_ring_filtered_values_hook(r, well_filter)

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

        well_filter = self._not_in_well_filter_hook(well_id)
        return self._get_values(well_filter, chunk_id)

    def get_val_values(well_id):
        well_filter = self._in_well_filter_hook(well_id)
        # chunk_id=0 to return all data
        return self._get_values(well_filter, chunk_id=0)

    def set_num_training_chunks(self, n_training_chunks):
        self._n_training_chunks = n_training_chunks
