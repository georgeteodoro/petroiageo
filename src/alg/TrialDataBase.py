import numpy as np
from abc import ABC, abstractmethod

import common


class TrialDataBase(ABC):
    '''
    Abstract trial_data class, which implements filtering and sampling. 
    All trial_data should be an array of the following columns, on this order:
     - x, y, z (point coordinates)
     - phi (porosity)
     - well_id (only required for feature selection, not propagation)
     - features (multiple columns)
    '''
    def __init__(self, n_features, features_only, wells_list, f_sel_filter,
                 porosity_data):
        # Set the datatype for points
        self._n_features = n_features
        self._features_only = features_only
        if features_only:
            self._base_data_type = [
                ('x', np.int64),
                ('y', np.int64),
                ('z', np.int64),
                ('phi', np.float64),
            ]
        else:
            self._base_data_type = [
                ('x', np.int64),
                ('y', np.int64),
                ('z', np.int64),
                ('phi', np.float64),
                ('well_id', np.int64),
            ]
        self._cur_data_type = self._base_data_type + [
            (f'f{f}', np.float64) for f in range(n_features)
        ]
        self._cur_data_type = np.dtype(self._cur_data_type)

        # Attributes for internal chunking, if necessary
        self._n_training_chunks = -1
        self._chunk_size = -1

        # Internal state
        self._current_feature_id = -1
        self._current_ring = -1
        self._current_features = []

        # Location of concrete trial_data. This should be initialize, accessed
        # and read through concrete backend subclass hooks. On the current
        # implementation, trial_data is a map of points per ring. Thus it is
        # easier to sample, reshape and add more points.
        self._trial_data_dict = dict()

        # # Current actual size of each ring. This size increases due to
        # # out-of-core porosity_data, which is ran through one chunk at a time
        # self._trial_data_size = dict()

        # Number of points within trial_data
        self._data_len = -1

        # This is the list of real wells coordinates
        self._wells_list = wells_list

        self._porosity_data = porosity_data
        self._f_sel_filter = f_sel_filter

    # =========================================================================
    # === Interface for subclasses ============================================
    # =========================================================================

    @abstractmethod
    def _set_ring_hook(self, ring, data):
        '''
        Should add data to a trial_data ring, updating internally its size.
        This method is called only once to add all points of a ring, for 
        all rings.
        '''
        raise Exception("[TrialDataBase][_append_ring_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _update_col_from_ring_hook(self, r, feature_data):
        '''
        Should update the last column of ring r.
        It is assumed that feature_data if already filtered for points on
        ring r. Thus, feature_data should have the correct size of the 
        internal trial_data for ring r.
        '''
        raise Exception("[TrialDataBase][_update_col_from_ring_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _get_ring_filtered_values_hook(self, r, well_filter):
        '''
        Should return a set of points for ring r, filtered by a well_filter.
        '''
        raise Exception("[TrialDataBase][_get_ring_filtered_values_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _in_well_filter_hook(self, well_id):
        '''
        Should return a function f(data) which filters data based on
        well_id and returns only the subset of data for which 
        data['well_id'] == well_id.
        '''
        raise Exception("[TrialDataBase][_in_well_filter_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _not_in_well_filter_hook(self, well_id):
        '''
        Should return a function f(data) which filters data based on
        well_id and returns only the subset of data for which 
        data['well_id'] != well_id. If well_id<0, should return the
        Whole data.
        '''
        raise Exception("[TrialDataBase][_not_in_well_filter_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _get_ring_np_values_hook(self, ring):
        '''
        Should return an nparray with all data from a given ring. It is ok
        do do such, since this method should only be used internally.
        '''
        raise Exception("[TrialDataBase][_get_ring_np_values_hook] "
                        "Abstract method not implemented.")

    # =========================================================================
    # === Public interface ====================================================
    # =========================================================================

    def prepare_porosity(self, it):
        '''
        Allocate data required for the current iteration it.
        Fill coordinates, phi and well_id (when necessary).
        It also resets the internal current column.
        First iteration is 1. Iteration 0 does not exists.
        '''

        # Reset internal state
        assert it>0, f"[TrialDataBase][prepare_porosity] "\
            f"First iteration is 1, but received current iteration {it}."
        self._current_ring = it - 1
        self._current_feature_id = 0
        self._current_features = []

        # Set ring to be filtered
        self._f_sel_filter.set_ring(self._current_ring)

        # Set the first feature, even if it's empty
        self._current_features.append(f'f{self._current_feature_id}')

        # Remove, if necessary, old data from previous rings
        # This should be done if sampling is required
        pass

        # Create new ring data
        self._trial_data_dict[self._current_ring] = []

        # Iterate on all porosity chunks to fill trial_data
        points_list = []
        for chunk_slice in self._porosity_data.iter_chunks():
            # Skip this chunk if there are not any points withing it
            if not common.has_points_within_chunk(
                    self._wells_list, self._current_ring, chunk_slice):
                continue

            # Load porosity data chunk
            chunk_np = self._porosity_data[chunk_slice]

            # Add points to temporary points_list
            filt_list = self._f_sel_filter.satisfies(chunk_np)
            filt_data = chunk_np[filt_list]

            if self._features_only:
                points_list.extend(filt_data[['x', 'y', 'z', 'phi']].tolist())
            else:
                points_list.extend(filt_data[[
                    'x',
                    'y',
                    'z',
                    'phi',
                    'well_id',
                ]].tolist())

        # Fill ring dict
        self._set_ring_hook(self._current_ring, points_list)

        # Update size and chunking info
        self._chunk_size = 0
        for r in self._trial_data_dict.keys():
            self._chunk_size += len(self._trial_data_dict[r])

    def commit_feature(self):
        '''
        Commits the current feature, then setting up the next feature.
        '''
        assert self._current_feature_id >= 0, "[TrialDataBase][commit_feature] "\
            "Committing feature before prepare_poroisity."
        assert self._current_feature_id < self._n_features, \
            "[TrialDataBase][commit_feature] Committing beyond last feature."

        self._current_feature_id += 1
        self._current_features.append(f'f{self._current_feature_id}')

    def update_feature(self, feature, disp, disp_cube_shape):
        '''
        Adds data to the last feature. Data is related to all rings.
        Receives a FeatureDataBase object and a displacement to apply on
        the input feature.
        '''

        # Fill data, one ring at a time
        for r in self._trial_data_dict.keys():
            # Retrieve the coordinate list and apply the feature displacement
            ring_coords = self._get_ring_np_values_hook(r)[['x', 'y',
                                                            'z']].copy()
            # This algorithm applies the displacement at the whole array,
            # allowing improved data access times
            for coord_s, d_id in [('x', 0), ('y', 1), ('z', 2)]:
                ring_coords[coord_s] = (ring_coords[coord_s] + disp[d_id] +
                                        ((disp_cube_shape[d_id] - 1) / 2))

            # Extract displaced feature data and assign it to the last col
            filtered_feature_data = feature.filter_coords(ring_coords)
            self._update_col_from_ring_hook(r, filtered_feature_data)

    def get_train_values(self, well_id, chunk_id):
        '''
        Leave-one-well-out validation function. Returns all data that
        is NOT on well_id. If well_id=-1, then all data is returned.
        Generates the training data inplace. X_train and y_train are
        generated inplace to avoid reallocation for them.
        If no out-of-core is used internally, then chunk_id=0 and all
        trial_data is returned (filtered by well_id obviously).
        '''

        well_filter = self._not_in_well_filter_hook(well_id)
        return self._get_values(well_filter, chunk_id)

    def get_val_values(self, well_id):
        well_filter = self._in_well_filter_hook(well_id)
        # chunk_id=0 to return all data
        return self._get_values(well_filter, chunk_id=0)

    def set_num_training_chunks(self, n_training_chunks):
        self._n_training_chunks = n_training_chunks

    def get_num_wells(self):
        return len(self._wells_list)

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

    def _get_values(self, well_filter, chunk_id):
        '''
        Helper function for filtering trial_data.
        Returns the number of filtered points.
        '''

        # Fill training data, one ring at a time
        X = []
        y = []
        for r in self._trial_data_dict.keys():

            # CHUNKING NOT IMPLEMENTED
            # Should be something like:
            # for each ring, return the ratio of |ring|/|trial_data|
            # with the proper range
            if chunk_id > 0:
                raise Exception("[TrialDataBase][_get_values] Chunking "
                                "not implemented for incremental learning.")

            new_points = self._get_ring_filtered_values_hook(r, well_filter)

            # Split X from y
            new_points_X = new_points[self._current_features]
            new_points_y = new_points['phi']

            # Add them to output arrays
            X.extend(new_points_X)
            y.extend(new_points_y)

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        X = np.array(np.array(X).tolist())
        y = np.array(np.array(y).tolist())

        return X, y
