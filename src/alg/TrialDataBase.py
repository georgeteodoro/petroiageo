import numpy as np
from abc import ABC, abstractmethod
from time import time

import common
from sampler import ChunkSamplerV1


class TrialDataBase(ABC):
    '''
    Abstract trial_data class, which implements filtering and sampling. 
    All trial_data should be an array of the following columns, on this order:
     - x, y, z (point coordinates)
     - phi (porosity)
     - well_id (only required for feature selection, not propagation)
     - features (multiple columns)
    Due to the use of padding, all coordinates are the padded coordinates. 
    Thus, it is expected of the wells_list to have padded coordinates as well.
    '''
    def __init__(self, features_only, f_sel_filter, porosity_data, config):
        self._config = config

        self._n_features = config.alg['max_num_features']

        # Set the datatype for points
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
            (f'f{f}', np.float64) for f in range(self._n_features + 1)
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
        # self._data_len = -1

        # This is the list of real wells coordinates
        self._wells_list = config.train_wells_coords

        self._porosity_data = porosity_data
        self._f_sel_filter = f_sel_filter

        # Setup sampling, if required
        if (config.alg.get('sampling') == None or
                config.alg['sampling'].get('sampler') == None or
                config.alg['sampling']['sampler'] == 'none'):
            self._sampler = None
        elif config.alg['sampling']['sampler'] == 'v1':
            self._sampler = ChunkSamplerV1(config)
        else:
            self._sampler = None

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
    def _get_ring_filtered_values_hook(self, r, chunk_id, well_filter):
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

    def prepare_porosity(self, prep_it):
        '''
        Allocate data required for the current iteration it.
        Fill coordinates, phi and well_id (when necessary).
        It also resets the internal current column.
        First iteration is 1. Iteration 0 does not exists.
        '''

        profile = self._config.get_param('prof_trial_prep_porosity')
        rings_to_keep = self._config.alg['sampling']['layers_window_size']

        t0 = time()

        # Reset internal state
        assert prep_it>0, f"[TrialDataBase][prepare_porosity] "\
            f"First iteration is 1, but received current iteration {prep_it}."

        # Set the first feature, even if it's empty
        self._current_feature_id = 0
        self._current_features = []
        self._current_features.append(f'f{self._current_feature_id}')

        # Remove, if necessary, old data from previous rings
        # This should be done if sampling is required
        trial_keys = sorted(self._trial_data_dict.keys())
        if (rings_to_keep != None and rings_to_keep > 0 and
                len(trial_keys) == rings_to_keep):
            self._trial_data_dict.pop(trial_keys[0])

        # Load all rings if this is a continued iteration
        it_init = 0 if self._current_ring < 0 else prep_it - 1

        t1 = time()

        # Load rings
        for it in range(it_init, prep_it):

            t11 = time()

            # Set ring to be filtered
            self._f_sel_filter.set_ring(it)

            # Create new ring data
            self._trial_data_dict[it] = []

            # Iterate on all porosity chunks to fill trial_data
            points_list = []
            for chunk_slice in self._porosity_data.iter_chunks():
                t111 = time()

                # Skip this chunk if there are not any points withing it
                if not common.has_points_within_chunk(self._wells_list, it,
                                                      chunk_slice):
                    continue

                # Load porosity data chunk
                chunk_np = self._porosity_data[chunk_slice]

                t112 = time()

                # Add points to temporary points_list
                filt_list = self._f_sel_filter.satisfies(chunk_np)
                t113 = time()
                filt_data = chunk_np[filt_list]
                t114 = time()

                if self._features_only:
                    points_list.extend(filt_data[['x', 'y', 'z',
                                                  'phi']].tolist())
                else:
                    points_list.extend(filt_data[[
                        'x',
                        'y',
                        'z',
                        'phi',
                        'well_id',
                    ]].tolist())

                t115 = time()

                if profile:
                    print(f"[TrialDataBase][prepare_porosity] ring[{it+1}]"
                          f"chunk[{chunk_slice}] p_chunk_load: {t112-t111}")
                    print(f"[TrialDataBase][prepare_porosity] ring[{it+1}]"
                          f"chunk[{chunk_slice}] p_chunk_satisfy: {t113-t112}")
                    print(f"[TrialDataBase][prepare_porosity] ring[{it+1}]"
                          f"chunk[{chunk_slice}] p_chunk_filt: {t114-t113}")
                    print(f"[TrialDataBase][prepare_porosity] ring[{it+1}]"
                          f"chunk[{chunk_slice}] p_chunk_extend: {t115-t114}")

            t12 = time()

            # Fill ring dict
            self._set_ring_hook(it, points_list)

            # Update size and chunking info
            self._chunk_size = 0
            for r in self._trial_data_dict.keys():
                self._chunk_size += len(self._trial_data_dict[r])

            t13 = time()
            if profile:
                print(f"[TrialDataBase][prepare_porosity] ring[{it+1}] "
                      f"chunk_total: {t12-t11}")
                print(f"[TrialDataBase][prepare_porosity] ring[{it+1}] "
                      f"ring_update: {t13-t12}")

        self._current_ring = prep_it - 1

        t2 = time()
        if profile:
            print(f"[TrialDataBase][prepare_porosity] final_time {t2-t1}")

    def commit_feature(self):
        '''
        Commits the current feature, then setting up the next feature.
        '''
        assert self._current_feature_id >= 0, "[TrialDataBase][commit_feature] "\
            "Committing feature before prepare_porosity."
        assert self._current_feature_id < self._n_features, \
            "[TrialDataBase][commit_feature] Committing beyond last feature."

        self._current_feature_id += 1
        self._current_features.append(f'f{self._current_feature_id}')

    def update_feature(self, feature, disp):
        '''
        Adds data to the last feature. Data is related to all rings.
        Receives a FeatureDataBase object and a displacement to apply on
        the input feature.
        '''

        profile = self._config.get_param('prof_trial_update_feature')

        t1 = time()

        # Fill data, one ring at a time
        for r in self._trial_data_dict.keys():
            t11 = time()

            # Retrieve the coordinate list and apply the feature displacement
            ring_coords = self._get_ring_np_values_hook(r)[['x', 'y',
                                                            'z']].copy()
            # This algorithm applies the displacement at the whole array,
            # allowing improved data access times.
            # All coordinates are already padded
            for coord_s, d_id in [('x', 0), ('y', 1), ('z', 2)]:
                ring_coords[coord_s] = (ring_coords[coord_s] + disp[d_id])

            t12 = time()

            # Extract displaced feature data and assign it to the last col
            filtered_feature_data = feature.filter_coords(ring_coords)
            t13 = time()
            self._update_col_from_ring_hook(r, filtered_feature_data)
            t14 = time()

            if profile:
                print(f"[TrialDataBase][update_feature] ring[{r}] "
                      f"get_coords_disp: {t12-t11}")
                print(f"[TrialDataBase][update_feature] ring[{r}] "
                      f"filter_coords: {t13-t12}")
                print(f"[TrialDataBase][update_feature] ring[{r}] "
                      f"update_col: {t14-t13}")

        t2 = time()
        if profile:
            print(f"[TrialDataBase][update_feature] final_time: {t2-t1}")

    def get_train_values(self, well_id, chunk_id, it=-1, with_sampling=True):
        '''
        Leave-one-well-out validation function. Returns all data that
        is NOT on well_id. If well_id=-1, then all data is returned.
        Generates the training data inplace. X_train and y_train are
        generated inplace to avoid reallocation for them.
        If no out-of-core is used internally, then chunk_id=0 and all
        trial_data is returned (filtered by well_id obviously).
        '''

        well_filter = self._not_in_well_filter_hook(well_id)
        return self._get_values(well_filter, chunk_id, it, with_sampling)

    def get_val_values(self, well_id, it=-1, with_sampling=True):
        well_filter = self._in_well_filter_hook(well_id)
        # chunk_id=0 to return all data
        return self._get_values(well_filter, 0, it, with_sampling)

    def set_num_training_chunks(self, n_training_chunks):
        self._n_training_chunks = n_training_chunks

    def get_num_wells(self):
        return len(self._wells_list)

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

    def _get_values(self, well_filter, chunk_id, it, with_sampling):
        '''
        Helper function for filtering trial_data.
        Returns the number of filtered points.
        '''
        n_training_chunks = int(
            self._config.alg['parallel']['n_training_chunks'])
        chunk_size = sum([len(x) for x in self._trial_data_dict.values()])
        chunk_size //= n_training_chunks

        # Fill training data, one ring at a time
        X = []
        y = []
        for ring_key, ring in self._trial_data_dict.items():

            new_points = self._get_ring_filtered_values_hook(
                ring_key, chunk_id, well_filter)

            # Perform sampling. It should sample some of the points of a ring.
            # Combining all rings' points sampled, a chunk is formed.
            if with_sampling and self._sampler != None:
                new_points = self._sampler.sample(new_points, chunk_size, it)

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
