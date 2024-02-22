from abc import ABC, abstractmethod
from h5py import Dataset
import numpy as np
from time import time
from math import ceil

import common
from config_parser import Config
from data_filter import WellsSingleRingDataFilter
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

    def __init__(self, target_wells_ids_list, porosity_data: Dataset,
                 config: Config, should_consider_sampling:bool = True):

        self._config = config

        self._n_features = config.alg['max_num_features']

        # Set the datatype for points
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

        # Internal state
        self._current_feature_id = -1
        self._current_ring = -1
        self._current_features = []

        # Location of concrete trial_data. This should be initialized, accessed
        # and read through concrete backend subclass hooks.
        # Regardless of concrete backend implementation, trial_data is a
        # map of points per ring. Thus it is easier to sample, reshape and
        # add more points.
        # self._trial_data_dict = dict()

        # List of all rings' IDs which are present on the concrete backend
        # data structures. This allows concrete subclasses not to worry about
        # this common data, while this superclass can still know which rings
        # were successfully added to the concrete object.
        self._rings_list = []

        # This is the list target wells. These can be training or test wells.
        self._wells_id_list = target_wells_ids_list
        self._f_sel_filter = WellsSingleRingDataFilter(target_wells_ids_list)

        self._porosity_data = porosity_data

        # Setup sampling, if required
        self._sampler = None
        self._rings_to_keep = -1
        if config.alg.get('sampling') != None and should_consider_sampling:
            if config.alg['sampling'].get('sampler') != None and config.alg[
                    'sampling']['sampler'] == 'v1':
                self._sampler = ChunkSamplerV1(config)

            if config.alg['sampling'].get('layers_window_size') != None:
                self._rings_to_keep = config.alg['sampling'][
                    'layers_window_size']

    # =========================================================================
    # === Interface for subclasses ============================================
    # =========================================================================

    @abstractmethod
    def _new_ring_hook(self, ring):
        '''
        Create the backend representation of a new ring. If the ring already
        exists, it should raise an assertion exception.
        '''
        raise Exception("[TrialDataBase][_new_ring_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _clear_trial_data_hook(self):
        '''
        Should delete all trial data, in preparation for another iteration.
        '''
        raise Exception("[TrialDataBase][_clear_trial_data_hook] "
                        "Abstract method not implemented.")

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
    def _update_col_hook(self, r, w, feature_data):
        '''
        Should update the last column of ring r and well_id w.
        It is assumed that feature_data perfectly matches the data 
        for r and w. Thus, feature_data should have the correct size 
        of the internal data for ring r and well_id w.
        '''
        raise Exception("[TrialDataBase][_update_col_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _get_values_hook(self, ring, well_id, chunk_slice=None):
        '''
        Should return an nparray with all data for a given ring an well_id.
        Ideally, backend data should be stored separately, since this is a
        recurrent operation.
        The chunk_slice parameter allows the concrete class to better 
        implement its retrieval of data. If not used, all data is returned.
        '''
        raise Exception("[TrialDataBase][_get_values_hook] "
                        "Abstract method not implemented.")

    @abstractmethod
    def _well_size_hook(self, ring, well_id):
        '''
        Should return the number of points for a ring/well pair.
        '''
        raise Exception("[TrialDataBase][_well_size_hook] "
                        "Abstract method not implemented.")

    # =========================================================================
    # === Public interface ====================================================
    # =========================================================================

    def prepare_porosity(self, prep_it: int):
        '''
        Allocate data required for the current iteration it.
        Fill coordinates, phi and well_id (when necessary).
        It also resets the internal current column.
        First iteration is 1. Iteration 0 does not exists.
        '''

        profile = self._config.get_param('prof_trial_prep_porosity')

        t0 = time()

        assert prep_it>0, f"[TrialDataBase][prepare_porosity] "\
            f"First iteration is 1, but received current iteration {prep_it}."

        # Set the first feature, even if it's empty
        self._current_feature_id = 0
        self._current_features = []
        self._current_features.append(f'f{self._current_feature_id}')

        # Load all rings if this is a continued iteration
        it_init = 0 if self._current_ring < 0 else prep_it - 1

        # If the sampler is used, all rings data should be purged and
        # generated again. Also, only the last '_rings_to_keep' rings
        # are generated
        if self._sampler != None:

            if self._rings_to_keep > 0:
                it_init = max(prep_it - self._rings_to_keep, 0)
            else:
                it_init = 0

            # Remove all rings data
            self._rings_list.clear()
            self._clear_trial_data_hook()

        t1 = time()

        # Load rings
        for it in range(it_init, prep_it):

            t11 = time()

            # Set ring to be filtered
            self._f_sel_filter.set_ring(it)

            # Create new ring data
            self._rings_list.append(it)
            self._new_ring_hook(it)

            # Prepare a dict of points per well_id
            points_dict = {id: list() for id in self._wells_id_list}
            target_wells_coords = self._config.get_coords_of_target_wells_ids(
                self._wells_id_list)

            # Iterate on all porosity chunks to fill trial_data
            for chunk_slice in self._porosity_data.iter_chunks():
                t111 = time()

                # Skip this chunk if there are not any points withing it
                if not common.has_points_within_chunk(target_wells_coords, it,
                                                      chunk_slice):
                    continue

                # Load porosity data chunk
                chunk_np = self._porosity_data[chunk_slice]

                t112 = time()

                # Add points to temporary points_dict
                filt_list = self._f_sel_filter.satisfies(chunk_np)
                t113 = time()
                filt_data = chunk_np[filt_list]
                t114 = time()

                # Extend points_dict by each well_id
                for w in self._wells_id_list:
                    well_data = filt_data[filt_data['well_id'] == w]
                    well_data = well_data[['x', 'y', 'z', 'phi', 'well_id']]

                    points_dict[w].extend(well_data.tolist())

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
            # REFACTORING/OPTIMIZARION OPORTUNITY:
            # Change _set_ring_hook to _append_ring_hook, thus points_dict is
            # not required. I.e., less memory needed. For numpy implementation
            # a temporary list may still be required within it, which is
            # converted to ndarray at the first access.
            self._set_ring_hook(it, points_dict)

            t13 = time()
            if profile:
                print(f"[TrialDataBase][prepare_porosity] ring[{it+1}] "
                      f"chunk_total: {t12-t11}")
                print(f"[TrialDataBase][prepare_porosity] ring[{it+1}] "
                      f"ring_update: {t13-t12}")

        self._current_ring = prep_it - 1

        # Check if it is necessary to perform sampling
        if self._sampler != None:
            self._perf_sampling(it + 1)

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
        if self._current_feature_id < self._n_features:
            self._current_features.append(f'f{self._current_feature_id}')

    def update_feature(self, feature, disp):
        '''
        Adds data to the last feature. Data is related to all rings.
        Receives a FeatureDataBase object and a displacement to apply on
        the input feature.

        Data to update the backend structure is added by both 
        ring and well_id. This allows reduced memory requirements
        since data is retrieved slowly. TrialDataBase doesn't care how 
        the data is stored, as long as it can access it by ring and well_id.

        Observation: This is not memory-optimized. For a later iteration,
        with 3 ring and 10 wells, for instance, the maximum amount of data 
        being loaded to memory is the size of a ring/well set of points.
        One way to reduce this is through chunked update
        '''

        profile = self._config.get_param('prof_trial_update_feature')

        t1 = time()

        # Fill data, one ring at a time
        for r in self._rings_list:
            for w in self._wells_id_list:

                # Retrieve the coordinate list of the
                cur_coords = self._get_values_hook(r, w)[['x', 'y', 'z']].copy()
                # Applies the displacement at the whole array,
                # allowing improved data access times.
                # No padding resolution is required since
                # all coordinates are already padded.
                for coord_s, d_id in [('x', 0), ('y', 1), ('z', 2)]:
                    cur_coords[coord_s] = (cur_coords[coord_s] + disp[d_id])

                # Extract displaced feature data
                filtered_feature_data = feature.filter_coords(cur_coords)

                # Assign feature data to the last col (i.e., current col
                # being updated)
                self._update_col_hook(r, w, filtered_feature_data)

                # t11 = time()
                # t12 = time()

                # t13 = time()
                # t14 = time()

                # if profile:
                #     print(f"[TrialDataBase][update_feature] ring[{r}] "
                #           f"get_coords_disp: {t12-t11}")
                #     print(f"[TrialDataBase][update_feature] ring[{r}] "
                #           f"filter_coords: {t13-t12}")
                #     print(f"[TrialDataBase][update_feature] ring[{r}] "
                #           f"update_col: {t14-t13}")

        t2 = time()
        if profile:
            print(f"[TrialDataBase][update_feature] final_time: {t2-t1}")

    def get_train_values(self, well_id, chunk_id):
        '''
        Leave-one-well-out validation function. Returns all data that
        is NOT on well_id. If well_id=-1, then all data is returned.
        Generates the training data inplace. X_train and y_train are
        generated inplace to avoid reallocation for them.
        If no out-of-core is used internally, then chunk_id=0 and all
        trial_data is returned (filtered by well_id obviously).
        '''

        # wells_to_retrieve is a list of indices
        wells_to_retrieve = list(self._wells_id_list)
        if well_id >= 0:
            wells_to_retrieve.remove(well_id)
        return self._get_values(wells_to_retrieve, chunk_id)

    def get_val_values(self, well_id):
        wells_to_retrieve = [well_id]
        # chunk_id=0 to return all data
        return self._get_values(wells_to_retrieve, -1)

    def get_num_wells(self):
        return len(self._wells_id_list)

    def _ring_size(self, ring):
        length = 0
        for w in self._wells_id_list:
            length += self._well_size_hook(ring, w)
        return length

    def __len__(self):
        length = 0
        for r in self._rings_list:
            length += self._ring_size(r)
        return length

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

    def _get_values(self, wells_to_retrieve, chunk_id):
        '''
        Helper function for filtering trial_data.
        Data is retrieved by ring and well_id until a chunk is reached.
        '''

        n_training_chunks = int(
            self._config.alg['parallel']['n_training_chunks'])

        # Fill training data, one ring at a time, one well at a time
        X = []
        y = []
        for r in self._rings_list:
            for w in wells_to_retrieve:
                # If chunking is used (i.e., not validation or test data)
                if chunk_id >= 0:
                    # Calculate how many points from a ring/well_id pair
                    # this chunk should have
                    points_per_well = self._well_size_hook(r, w)
                    points_per_rw = int(
                        ceil(points_per_well / n_training_chunks))

                    # Generate a chunk slice for the ring/well pair
                    beg = chunk_id * points_per_rw
                    end = (chunk_id + 1) * points_per_rw
                    end = min(end, points_per_well)
                    cur_slice = chunk_slice = slice(int(beg), int(end))
                else:
                    cur_slice = chunk_slice = slice(0,
                                                    self._well_size_hook(r, w))

                # Retrieve current chunk slice from the backend storage
                new_points = self._get_values_hook(r, w, cur_slice)

                # Assuming that new_points is a np.ndarray
                if new_points.size > 0:
                    # Split X from y
                    new_points_X = new_points[self._current_features]
                    new_points_y = new_points['phi']
                else:
                    new_points_X = list()
                    new_points_y = list()

                # Add them to output arrays
                X.extend(new_points_X)
                y.extend(new_points_y)

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        # This is required for lgb.train. It can't receive structured ndarray
        # only a regular array
        # TODO: Tentar melhorar isso
        X = np.array(np.array(X).tolist())
        y = np.array(np.array(y).tolist())

        return X, y

    def _perf_sampling(self, it):
        # Number of all points in
        total_n_points = len(self)

        # Sample each ring individually
        # TODO: Sampling is memory inefficient: all data from a given ring is
        # first compiled and then sampled. Maybe later change the sampler to
        # receive as input points from a ring/well pair.
        for ring in self._rings_list:
            # print(f'================= ring{ring}')
            # Compile all points from a ring
            ring_points = []
            for well_id in self._wells_id_list:
                ring_points.extend(self._get_values_hook(ring, well_id))

            # Perform sampling
            new_ring_points = self._sampler.sample(np.array(ring_points),
                                                   total_n_points, it, ring)

            # Split all points by well_id and add them to a dict
            new_rings_dict = dict()
            field_names = [i for i, j in self._base_data_type]
            for well_id in self._wells_id_list:
                new_rings_dict[well_id] = new_ring_points[
                    new_ring_points['well_id'] == well_id][field_names]

            # Update the internal concrete data with the sampled points
            self._set_ring_hook(ring, new_rings_dict)

            # # Used for getting the sampled coords for
            # # sampling integration testing.
            # print(f'------------------- ring{ring_key}:')
            # print(new_ring[['x', 'y', 'z']])
