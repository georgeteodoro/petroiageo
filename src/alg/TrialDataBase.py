from abc import ABC, abstractmethod
from h5py import Dataset
from mpi4py import MPI
import numpy as np
from numpy.lib import recfunctions as rfn
from time import time
from math import ceil
from collections import defaultdict
from decimal import Decimal

import common
from config_parser import Config
from data_filter import WellsSingleRingDataFilter
from sampler import ChunkSamplerV1, target_based_sampler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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

    def __init__(self,
                 target_wells_ids_list: list[int],
                 porosity_data: Dataset,
                 config: Config,
                 should_consider_sampling: bool = True):

        self._config = config

        self._n_features = config.alg['max_num_features']

        # Set the datatype for points
        self._base_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('well_id', np.int64),
            ('real', np.int64),
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
            elif config.alg['sampling'].get('sampler') != None and config.alg[
                    'sampling']['sampler'] == 'v2':
                self._sampler = 'v2'

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
    def _set_ring_hook(self, ring, data, overwrite=False):
        '''
        Should add data to a trial_data ring, updating internally its size.
        This method is called only once to add all points of a ring, for 
        all rings.

        data should be a dict of points grouped by well_id.
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
        First iteration is 0.

        prep_it: Integer representing the iteration whose data we should 
                 prepare. For running iteration 20 (will propagate ring 20)
                 prep_it should be 20.
        '''
        profile = self._config.get_param('prof_trial_prep_porosity')

        t0 = time()

        # Retirar esse assert ou alterar para >=0
        assert prep_it>=0, f"[TrialDataBase][prepare_porosity] "\
            f"First iteration is 0, but received current iteration {prep_it}."

        # Set the first feature, even if it's empty
        self._current_feature_id = 0
        self._current_features = []
        self._current_features.append(f'f{self._current_feature_id}')

        # Load all rings if this is a continued iteration
        # As each iteration it propagates the ring = it, at the start of an it
        # we should load the ring propagated in the last it, that is
        # prep_it - 1 if this is a continued iteration. Otherwise, load all.
        # This logic only applies if each it propagates only 1 ring
        start_ring_idx_to_load = 0 if self._current_ring < 0 else prep_it - 1

        # If the sampler is used, all rings data should be purged and
        # generated again. Also, only the last '_rings_to_keep' rings
        # are generated
        if self._sampler != None:

            # TODO: This should be checked only if the sampler is 'v1'
            if self._rings_to_keep > 0:
                start_ring_idx_to_load = max(prep_it - self._rings_to_keep, 0)
            # TODO: This else should be elif self._current_ring < 0, otherwise,
            # in continued iteration, even if we already loaded the previous
            # rings, we would load them again. This should be done only if it
            # is not a continued iteration
            else:
                start_ring_idx_to_load = 0

            # Remove all rings data
            self._rings_list.clear()
            self._clear_trial_data_hook()

            # TODO: deprecate _clear_trial_data_hook() since it is the
            # sampler's job to define which data should be available
        t1 = time()

        # Load rings, one at a time
        # Suposes that a it only propagates one ring
        end_ring_idx_to_load = prep_it if prep_it > 0 else 1
        for ring_to_load in range(start_ring_idx_to_load, end_ring_idx_to_load):
            t11 = time()

            # Set ring to be filtered
            self._f_sel_filter.set_ring(ring_to_load)
            # Create new ring data
            self._rings_list.append(ring_to_load)
            self._new_ring_hook(ring_to_load)
            # Prepare a dict of points per well_id
            points_dict = {wid: list() for wid in self._wells_id_list}
            target_wells_coords = self._config.get_coords_of_target_wells_ids(
                self._wells_id_list)
            # Iterate on all porosity chunks to fill trial_data
            for chunk_slice in self._porosity_data.iter_chunks():
                t111 = time()

                # Skip this chunk if there are not any points withing it
                if not common.has_points_within_chunk(
                        target_wells_coords, ring_to_load, chunk_slice):
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
                    well_data = well_data[[
                        'x', 'y', 'z', 'phi', 'well_id', 'real'
                    ]]

                    points_dict[w].extend(well_data.tolist())

                t115 = time()

                if profile:
                    print(
                        f"[TrialDataBase][prepare_porosity] ring[{ring_to_load}]"
                        f"chunk[{chunk_slice}] p_chunk_load: {t112-t111:.5f}")
                    print(
                        f"[TrialDataBase][prepare_porosity] ring[{ring_to_load}]"
                        f"chunk[{chunk_slice}] p_chunk_satisfy: {t113-t112:.5f}"
                    )
                    print(
                        f"[TrialDataBase][prepare_porosity] ring[{ring_to_load}]"
                        f"chunk[{chunk_slice}] p_chunk_filt: {t114-t113:.5f}")
                    print(
                        f"[TrialDataBase][prepare_porosity] ring[{ring_to_load}]"
                        f"chunk[{chunk_slice}] p_chunk_extend: {t115-t114:.5f}")

            t12 = time()

            # print(f'=========== points_dict.shape for ring {it}: '
            #       f'{[(wid, len(d)) for wid, d in points_dict.items()]}')

            # Fill ring dict
            # REFACTORING/OPTIMIZATION OPORTUNITY:
            # Change _set_ring_hook to _append_ring_hook, thus points_dict is
            # not required. I.e., less memory needed. For numpy implementation
            # a temporary list may still be required within it, which is
            # converted to ndarray at the first access.
            self._set_ring_hook(ring_to_load, points_dict)

            t13 = time()
            if profile:
                print(f"[TrialDataBase][prepare_porosity] ring[{ring_to_load}] "
                      f"chunk_total: {t12-t11:.4f}")
                print(f"[TrialDataBase][prepare_porosity] ring[{ring_to_load}] "
                      f"ring_update: {t13-t12:.4f}")

        # Check if it is necessary to perform sampling
        if self._sampler != None:
            self._perf_sampling(prep_it)

        self._current_ring = prep_it

        t2 = time()
        if profile:
            print(f"[TrialDataBase][prepare_porosity] final_time {t2-t1:.4f}")

    def commit_feature(self, feature, disp):
        '''
        Commits the current feature, then setting up the next feature.
        '''
        assert self._current_feature_id >= 0, "[TrialDataBase][commit_feature] "\
            "Committing feature before prepare_porosity."
        assert self._current_feature_id < self._n_features, \
            "[TrialDataBase][commit_feature] Committing beyond last feature: "\
            f"cur_feature={self._current_feature_id} "\
            f"n_features={self._n_features}."

        # Hook used by concurrent implementations of TrialDataBase.
        # Default behavior is: return True, i.e., all processes perform
        # update_feature().
        should_commit_feature = self._should_commit_feature_hook()
        if should_commit_feature:
            self.update_feature(feature, disp)
            self._done_commit_feature_hook()

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

        get_coords_disp_time = 0
        apply_disp_time = 0
        filter_coords_time = 0
        update_col_time = 0

        #Isso está certo
        # Fill data, one ring at a time
        for r in self._rings_list:
            for w in self._wells_id_list:
                t11 = time()

                # Check if there are points for a ring/well pair. Example for
                # when there are no points: well w has already propagated
                # all it could, being surrounded by other wells, thus there
                # may be a ring for which it has no points.
                cur_values = self._get_values_hook(r, w)
                if len(cur_values) == 0:
                    continue

                # Retrieve the coordinate list of the current ring/well pair
                cur_coords = cur_values[['x', 'y', 'z']]
                t12 = time()

                # Applies the displacement at the whole array,
                # allowing improved data access times.
                # No padding resolution is required since
                # all coordinates are already padded.
                for coord_s, d_id in [('x', 0), ('y', 1), ('z', 2)]:
                    cur_coords[coord_s] = (cur_coords[coord_s] + disp[d_id])
                t13 = time()

                # Extract displaced feature data
                filtered_feature_data = feature.filter_coords(cur_coords)
                t14 = time()

                # Assign feature data to the last col (i.e., current col
                # being updated)
                self._update_col_hook(r, w, filtered_feature_data)
                t15 = time()

                get_coords_disp_time += t12 - t11
                apply_disp_time += t13 - t12
                filter_coords_time += t14 - t13
                update_col_time += t15 - t14

                # if profile:
                #     print(f"[TrialDataBase][update_feature][r{r}][w{w}] "
                #           f"get_coords_disp: {t12-t11:.4f}")
                #     print(f"[TrialDataBase][update_feature][r{r}][w{w}] "
                #           f"apply_disp: {t13-t12:.4f}")
                #     print(f"[TrialDataBase][update_feature][r{r}][w{w}] "
                #           f"filter_coords: {t14-t13:.4f}")
                #     print(f"[TrialDataBase][update_feature][r{r}][w{w}] "
                #           f"update_col: {t15-t14:.4f}")

        t2 = time()
        if profile:
            print(f"[TrialDataBase][update_feature] final_time: {t2-t1:.4f}")
            print(f"[TrialDataBase][update_feature] "
                  f"get_coords_disp: {get_coords_disp_time:.2f}")
            print(f"[TrialDataBase][update_feature] "
                  f"apply_disp: {apply_disp_time:.2f}")
            print(f"[TrialDataBase][update_feature] "
                  f"filter_coords: {filter_coords_time:.2f}")
            print(f"[TrialDataBase][update_feature] "
                  f"update_col: {update_col_time:.2f}")

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

        n_training_chunks = self._config.get_param('n_training_chunks')
        profile = self._config.get_param('prof_TD_get_values')
        sequential_chunking = self._config.get_param('sequential_chunking')

        prep_slice_time = 0
        get_val_hook_time = 0
        to_list_time = 0
        append_time = 0

        # Output collection of data. Each data chunk (ring/well pair) is
        # appended to the lists bellow. Later these are concatenated, avoiding
        # using extend and reallocating data.
        X = []
        y = []

        def _update_X_Y(new_points, cur_features, is_val, X, Y):
            # Assuming that new_points is a np.ndarray
            if new_points.size > 0:
                # Only real points should be used for validation
                if is_val:
                    new_points = new_points[new_points['real'] ==
                                            common.RealValues.real]

                # Split X from y
                new_points_X = new_points[cur_features]
                new_points_y = new_points['phi']

                # This conversion removes the structured array information,
                # converting to a simple 2D ndarray (lines, fields). Now
                # the conversion does not require expensive copying/moving
                # the whole data points multiple times just to be
                # compatible with lgb.train().
                new_points_X = rfn.structured_to_unstructured(new_points_X)

                # t3 = time()
                # to_list_time += t3 - t2

                # Add them to output arrays
                X.append(new_points_X)
                y.append(new_points_y)

                # t4 = time()
                # append_time += t4 - t3

        # Sequential chunking is the simplest solution if there is no
        # hierarchical storage (ring,well). This should not be used,
        # only here for publication experiments.
        if sequential_chunking and chunk_id >= 0:
            # Total number of points is related to the actual points for
            # wells_to_retrieve
            wells_len = 0
            for r in self._rings_list:
                for w in wells_to_retrieve:
                    wells_len += self._well_size_hook(r, w)

            points_per_chunk = int(ceil(wells_len / n_training_chunks))
            cur_chunk_id = 0
            cur_chunk_len = 0
            prev_rw = None
            prev_beg = -1
            prev_remaining = 0
            for r in self._rings_list:
                for w in wells_to_retrieve:

                    # Chunk was assembled
                    if cur_chunk_len >= points_per_chunk:
                        if cur_chunk_id == chunk_id:
                            # The expected chunk was filled, then nothing
                            # else to do
                            break
                        else:
                            # Resets the chunk and try to get a new one
                            cur_chunk_id += 1
                            cur_chunk_len = 0

                    # First, check if the previous ring,well have some
                    # leftover points. We assume that it is IMPOSSIBLE
                    # for a ring,well array to be larger than
                    # two points_per_chunk.
                    if prev_rw is not None:
                        cur_chunk_len += prev_remaining
                        cur_slice = slice(prev_beg, None)

                        # All chunks are symbolically filled until the expected
                        # chunk needs to be filled. Only then data is read.
                        if cur_chunk_id == chunk_id:
                            # Retrieve current chunk slice from the backend storage
                            t1 = time()
                            new_points = self._get_values_hook(
                                *prev_rw, cur_slice)

                            t2 = time()
                            get_val_hook_time += t2 - t1

                            # Update X and Y with the new points to be returned for
                            # the current chunk_id
                            _update_X_Y(new_points, self._current_features,
                                        len(wells_to_retrieve) == 1, X, y)

                        # Reset prev ring,well with points
                        prev_remaining = 0
                        prev_beg = -1
                        prev_rw = None

                    # Trying to add the current ring,well
                    rw_len = self._well_size_hook(r, w)

                    if cur_chunk_len + rw_len <= points_per_chunk:
                        # The current ring,well is not enough for the
                        # full chunk. Then add it entirely.

                        cur_chunk_len += rw_len

                        # Generate a chunk slice for the ring/well pair
                        # with all points
                        cur_slice = slice(0, rw_len)
                    else:
                        # The current ring,well overfills the full chunk.
                        # Then only add enough.
                        expected_len = points_per_chunk - cur_chunk_len

                        cur_chunk_len += expected_len
                        cur_slice = slice(0, expected_len)

                        prev_beg = expected_len
                        prev_remaining = rw_len - expected_len
                        prev_rw = (r, w)

                    # All chunks are symbolically filled until the expected
                    # chunk needs to be filled. Only then data is read.
                    if cur_chunk_id == chunk_id:
                        # Retrieve current chunk slice from the backend storage
                        t1 = time()
                        new_points = self._get_values_hook(r, w, cur_slice)

                        t2 = time()
                        get_val_hook_time += t2 - t1

                        # Update X and Y with the new points to be returned for
                        # the current chunk_id
                        _update_X_Y(new_points, self._current_features,
                                    len(wells_to_retrieve) == 1, X, y)

        else:
            # Fill training data, one ring at a time, one well at a time
            for r in self._rings_list:
                for w in wells_to_retrieve:
                    t0 = time()
                    # If chunking is used (i.e., not validation or test data)
                    if chunk_id >= 0:
                        # Calculate how many points from a ring/well_id pair
                        # this chunk should have
                        rw_len = self._well_size_hook(r, w)
                        points_per_rw = int(ceil(rw_len / n_training_chunks))

                        # Generate a chunk slice for the ring/well pair
                        beg = chunk_id * points_per_rw
                        end = (chunk_id + 1) * points_per_rw
                        end = min(end, rw_len)
                        cur_slice = slice(int(beg), int(end))
                    else:
                        cur_slice = slice(0, self._well_size_hook(r, w))
                    t1 = time()
                    prep_slice_time += t1 - t0

                    # Retrieve current chunk slice from the backend storage
                    new_points = self._get_values_hook(r, w, cur_slice)

                    t2 = time()
                    get_val_hook_time += t2 - t1

                    # Update X and Y with the new points to be returned for
                    # the current chunk_id
                    _update_X_Y(new_points, self._current_features,
                                len(wells_to_retrieve) == 1, X, y)

        # Concatenate all temporary arrays into a single output array
        t5 = time()
        if len(X) > 0:
            X = np.concatenate(X)
            y = np.concatenate(y)
        t6 = time()

        if profile:
            print(f"[TrialDataBase][_get_values] prep_slice "
                  f"{prep_slice_time:.4f}")
            print(f"[TrialDataBase][_get_values] "
                  f"_get_values_hook {get_val_hook_time:.4f}")
            # print(f"[TrialDataBase][_get_values] to_list {to_list_time:.4f}")
            # print(f"[TrialDataBase][_get_values] append {append_time:.4f}")
            print(f"[TrialDataBase][_get_values] concatenate {t6-t5:.4f}")

        return X, y

    def _perf_sampling_v2(self, it: int):
        """
        Do the target based sampling. Modifies internal state inplace
        """
        # Points of a bucket are: buckets_len[p] => poros of [p, p+width)
        # Values are Decimal to escape from python float imprecision, which
        # would screw the buckets indexing system. Also, must use string as
        # input instead of a number, which would be converted to float.
        poros_width = Decimal(str(self._config.alg['sampling']['poros_width']))
        poros_min = Decimal(str(self._config.alg['sampling']['poros_min']))
        poros_max = Decimal(str(self._config.alg['sampling']['poros_max']))

        samp_debug = True

        # list of all available buckets to fit porosity points
        buckets_list = [
            x * poros_width + poros_min
            for x in range(int((poros_max - poros_min) / poros_width))
        ]

        # List of rings which will be sampled. We assume that the first
        # ring was 'already sampled'. However, we don't sample it since
        # later sampling passes will shrink the first ring
        if it == 0:
            rings_to_sample = list()
        elif it == 1:
            rings_to_sample = [0]
        else:
            rings_to_sample = self._rings_list.copy()

        if samp_debug:
            print("[_perf_sampling_v2]Buckets list", buckets_list)
            print("[_perf_sampling_v2]Rings list:", self._rings_list)
            print('[_perf_sampling_v2]Rings to sample', rings_to_sample)

        buckets_max_size = self._config.alg['sampling']['bucket_max_size']
        # For logging purposes
        if samp_debug:
            self._log_train_data_buckets_size(it,
                                              poros_width,
                                              buckets_list,
                                              rings_to_sample,
                                              buckets_max_size,
                                              starting=True)

        rng = np.random.default_rng(seed=self._config.alg['sampling']['seed'])
        for ring_being_sampled in rings_to_sample:
            self._sample_from_ring(poros_width, samp_debug, buckets_list,
                                   ring_being_sampled, rng)

        # For logging purposes
        self._log_train_data_buckets_size(it,
                                          poros_width,
                                          buckets_list,
                                          rings_to_sample,
                                          buckets_max_size,
                                          starting=False)

    def _log_train_data_buckets_size(self, it, poros_width, buckets_list,
                                     rings_to_sample, buckets_max_size,
                                     starting: bool):
        rank_should_propagate = self._config.get_param(
            'mpi_should_update_local')
        if rank_should_propagate:
            starting_ring = 0
            if len(rings_to_sample) == 0:
                last_ring_to_count = 1
            elif starting:
                last_ring_to_count = rings_to_sample[-1]
            else:
                last_ring_to_count = rings_to_sample[-1] + 1

            target_rings = list(range(starting_ring, last_ring_to_count))
            beg_str = f"[it{it}][buckets_len]"
            beg_str += '[starting_buckets_len]' if starting else '[ending_buckets_len]'

            print(beg_str, f"Counting buckets from rings:", target_rings)
            buckets_len, _ = self._get_buckets_len(poros_width, buckets_list,
                                                   target_rings)

            print(beg_str, buckets_len)
            total_points = sum(buckets_len.values())
            print(beg_str, "Total points:", total_points)
            buckets_with_more_than_max_size = {
                key: value
                for key, value in buckets_len.items()
                if value > buckets_max_size
            }
            print(beg_str, "Buckets over max size:",
                  buckets_with_more_than_max_size)

    def _sample_from_ring(self,
                          poros_width: Decimal,
                          samp_debug: bool,
                          buckets_list: list[Decimal],
                          ring_being_sampled: int,
                          rng: np.random.Generator = None):
        """
        Do the v2 sampling for a ring and update internal data to take
        into account the sampled data. m

        Args:
        poros_width: The width of every bucket in porosity measurements
        samp_debug: Flag indicating we should print more stuff
        ring_being_sampled: The current ring being sampled
        buckets_list: List of Decimal indicating the starting value for every bucket
        rng: Numpy RandomNumberGenerator. Default is None. In this case, one is created
            using the sampling.seed from the config file
        """
        if samp_debug:
            print(f'=============== sampling ring {ring_being_sampled}')

        # List of rings from which points can be updated. E.g., remove
        # points when this ring needs to add some.
        # All rings, except the last one (maybe recently propagated) should
        # be available for shrinking
        this_ring_available_rings_to_shrink = list(range(0, ring_being_sampled))
        buckets_len, buckets_origin = self._get_buckets_len(
            poros_width, buckets_list, this_ring_available_rings_to_shrink)

        shuffled_wells_ids = self._wells_id_list.copy()
        if rng is None:
            rng = np.random.default_rng(
                seed=self._config.alg['sampling']['seed'])
        rng.shuffle(shuffled_wells_ids)
        # The sampling of 'ring_being_sampled' is done one well at a time
        # to reduce the memory footprint
        updated_ring_well_pairs = list()
        for well_id in shuffled_wells_ids:
            self._sample_from_well(poros_width, samp_debug, ring_being_sampled,
                                   buckets_len, buckets_origin,
                                   updated_ring_well_pairs, well_id, rng)

    def _sample_from_well(self,
                          poros_width: Decimal,
                          samp_debug: bool,
                          ring_being_sampled: int,
                          buckets_len: dict[Decimal, int],
                          buckets_origin: dict[Decimal, list[tuple[int, int]]],
                          updated_ring_well_pairs: list,
                          well_id: int,
                          rng: np.random.Generator = None):
        """
        Do the v2 sampling for a well in a ring. Updates internal state to take into
        account the sampled data. Modify updated_ring_well_pairs and
        buckets_len inplace

        Args:
        poros_width: The width of every bucket in porosity measurements
        samp_debug: Flag indicating we should print more stuff
        ring_being_sampled: The current ring being sampled
        buckets_len: Dict of length of buckets from rings previous to the 
            ring_being_sampled
        buckets_origin: Dict of list of bucket origins from rings previous to the 
            ring_being_sampled
        updated_ring_well_pairs: List of new (ring, well) pairs to consider points from
        well_id: Int representing the current well being sampled
        rng: Numpy RandomNumberGenerator. Default is None. In this case, one is created
            using the sampling.seed from the config file
        """
        alpha = self._config.alg['sampling']['alpha']
        bucket_max_size = self._config.alg['sampling']['bucket_max_size']
        if rng is None:
            rng = np.random.default_rng(
                seed=self._config.alg['sampling']['seed'])
        # Base columns names, e.g., x,y,x,phi,...
        field_names = [i for i, j in self._base_data_type]

        prop_points = self._get_values_hook(ring_being_sampled,
                                            well_id)[field_names]

        # Perform sampling to find out which points should remain
        points_to_add, points_to_remove = target_based_sampler(
            prop_points, buckets_len, poros_width, bucket_max_size, alpha, rng)

        if samp_debug:
            print(
                f'sampling r{ring_being_sampled}w{well_id}: '
                f'add {len(points_to_add)} rem {sum(points_to_remove.values())} '
                f'from {len(prop_points)} prop_points')

        # Update the ring,well data for 'ring_being_sampled'
        updated_dict = {well_id: points_to_add}
        self._set_ring_hook(ring_being_sampled, updated_dict, True)

        updated_ring_well_pairs.append((ring_being_sampled, well_id))

        removed_p_per_bucket = self._shrink_rings(
            poros_width, samp_debug, ring_being_sampled, bucket_max_size,
            buckets_origin, updated_ring_well_pairs, points_to_remove)

        self._remove_remaining_points(poros_width, samp_debug,
                                      ring_being_sampled, well_id,
                                      points_to_remove, removed_p_per_bucket)

    def _remove_remaining_points(self, poros_width: int, samp_debug: bool,
                                 ring_being_sampled: int, well_id: int,
                                 points_to_remove: dict[Decimal, int],
                                 removed_p_per_bucket: dict[Decimal, int]):
        """
        Remove points from the (well_id, ring_being_sampled) as needed
        """
        still_should_remove: dict = {
            key: (value - removed_p_per_bucket[key])
            for key, value in points_to_remove.items()
            if (value - removed_p_per_bucket[key]) > 0
        }

        if samp_debug:
            print(f"--- Still should remove "\
                  f"{sum(still_should_remove.values())} from the new"\
                    f" points of r{ring_being_sampled}w{well_id}")

        # Base columns names, e.g., x,y,x,phi,...
        field_names = [i for i, j in self._base_data_type]
        added_points = self._get_values_hook(ring_being_sampled,
                                             well_id)[field_names]
        for bucket, n_to_rem in still_should_remove.items():
            if n_to_rem == 0:
                continue

            within_bucket_cond = ((added_points['phi'] >= bucket)
                                  &
                                  (added_points['phi'] < bucket + poros_width))
            filt_points = added_points[within_bucket_cond]
            remaining_points = added_points[~within_bucket_cond]

            if samp_debug:
                print(
                    f"--- Removing {n_to_rem}/{len(filt_points)} from "\
                        f"r{ring_being_sampled}w{well_id} "\
                        f"bucket {bucket} to compensate"
                )

            assert_msg = f"[WARNING] should still remove {n_to_rem} from"\
                       f"r{ring_being_sampled}w{well_id} "\
                        f"but only got {len(filt_points)} points!"
            assert len(filt_points) >= n_to_rem, assert_msg

            np.random.shuffle(filt_points)
            # Update ring/well data
            sampled_points = np.concatenate(
                (filt_points[:-n_to_rem], remaining_points))

            updated_dict = {well_id: sampled_points}
            self._set_ring_hook(ring_being_sampled, updated_dict, True)

    def _shrink_rings(
            self, poros_width: Decimal, samp_debug: bool,
            ring_being_sampled: int, b_max_size: int,
            buckets_origin: dict[Decimal, list[tuple[int, int]]],
            updated_ring_well_pairs: list,
            points_to_remove: dict[Decimal, int]) -> dict[Decimal, int]:
        """
        Remove points from rings less or equal to ring_being_sampled
        defined from buckets_origin + updated_ring_well_pairs
        to balance the new sampled points on the v2 sampling.

        Returns a dict of how many points were removed per bucket
        """
        removed_p_per_bucket = dict()
        for bucket, n_to_rem_from_all_rings in points_to_remove.items():
            # Calculate how many points are within the current bucket
            tot_shrinkable_b_pts = self._calc_n_points_in_bucket_for_rings_leq_to(
                poros_width, ring_being_sampled,
                buckets_origin[bucket] + updated_ring_well_pairs, bucket)

            if samp_debug:
                # In theory, tot_shrinkable_b_pts < (b_max_size + n_to_rem_from_all_rings)
                # but, in practice, it might be >=.
                # TODO: solve this issue
                will_remove = tot_shrinkable_b_pts / (b_max_size +
                                                      n_to_rem_from_all_rings)
                will_remove *= n_to_rem_from_all_rings
                will_remove = int(will_remove)
                print(
                    f'++++ to_del {will_remove}/{tot_shrinkable_b_pts} points '
                    f'from bucket {bucket} as we want to remove {n_to_rem_from_all_rings} on total'
                )

            n_removed_points = 0
            # Shrink rings proportionally by 'n'
            if tot_shrinkable_b_pts > 0:
                pts_origin = buckets_origin[bucket] + updated_ring_well_pairs
                n_removed_points = self._shrink_bucket_proportionally(
                    poros_width, samp_debug, ring_being_sampled, b_max_size,
                    pts_origin, bucket, n_to_rem_from_all_rings)

            removed_p_per_bucket[bucket] = n_removed_points
            if samp_debug:
                print(
                    f"--- Wanted to remove {n_to_rem_from_all_rings}"\
                    f" and calculated to remove {n_removed_points}"
                )
        return removed_p_per_bucket

    def _shrink_bucket_proportionally(self, poros_width: Decimal,
                                      samp_debug: bool, max_ring: int,
                                      b_max_size: int,
                                      pts_origin: list[tuple[int, int]],
                                      bucket: Decimal,
                                      n_to_rem_from_all_rings: int) -> int:
        """
        Proportionally remove points from the bucket where points come from the pts_origin
        but aren't from rings greater then max_ring given the n_to_rem_from_all_rings.

        Return the number of points removed
        """
        n_removed_points = 0
        for ring, well_id in pts_origin:
            # Only account for removable points from the
            # less or equal rings. For a given ring_being_sampled=3
            # we should not remove points from ring 5 since it will
            # be sampled later.
            if ring > max_ring:
                continue

            filt_points, remaining_points = self._split_pts_in_and_out_of_bucket(
                poros_width, bucket, ring, well_id)

            # Calculate how many points should be removed. If none,
            # then just skip. This can only happen for rounding
            # n_to_rem to zero.
            current_tot_b_points = (b_max_size + n_to_rem_from_all_rings)
            n_to_rem = len(filt_points) / current_tot_b_points
            n_to_rem *= n_to_rem_from_all_rings
            n_to_rem = int(n_to_rem)

            if samp_debug:
                print(f'--- removing {n_to_rem}/{len(filt_points)} from '
                      f'r{ring}w{well_id}')

            if n_to_rem == 0:
                continue
                # Since only the last points are removed we shuffle
                # all points to avoid taking only consecutive points
            np.random.shuffle(filt_points)

            # Update ring/well data
            assert n_to_rem > 0, f"n_to_rem is {n_to_rem}, which doesnt makes sense!"
            sampled_points = np.concatenate(
                (filt_points[:-n_to_rem], remaining_points))
            updated_dict = {well_id: sampled_points}
            self._set_ring_hook(ring, updated_dict, True)

            # This is the true removed points
            n_removed_points += min(n_to_rem, len(filt_points))

        return n_removed_points

    def _split_pts_in_and_out_of_bucket(
            self, poros_width: Decimal, bucket: Decimal, ring: int,
            well_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Divide the points for a well and ring pair present and not in a bucket

        Args:
        poros_width: The width of the bucket
        bucket: The bucket porosity start
        ring: Target ring
        well_id: Target ring

        Returns the points that are in the bucket and those that aren't
        """
        field_names = [i for i, j in self._base_data_type]
        all_ring_well_points = self._get_values_hook(ring, well_id)[field_names]
        within_bucket_cond = (
            (all_ring_well_points['phi'] >= bucket)
            & (all_ring_well_points['phi'] < bucket + poros_width))
        in_bucket = all_ring_well_points[within_bucket_cond]
        out_of_bucket = all_ring_well_points[~within_bucket_cond]
        return in_bucket, out_of_bucket

    def _calc_n_points_in_bucket_for_rings_leq_to(self, poros_width: Decimal,
                                                  target_ring: int,
                                                  target_buckets_origin: list[
                                                      tuple[int, int]],
                                                  bucket: int) -> int:
        """
        Count how many points are within a bucket at rings less or equal to 
        the target_ring that are defined by the pairs at the target_buckets_origin

        Args:
        poros_width: The buckets width
        target_ring: The target ring
        target_buckets_origin: List of points origins made up of tuples of (ring, well_id)
        bucket: The bucket porosity start 
        """
        total_points = 0
        for ring, well_id in target_buckets_origin:
            # Only account for removable points
            if ring > target_ring:
                continue

            filt_points, _ = self._split_pts_in_and_out_of_bucket(
                poros_width, bucket, ring, well_id)
            total_points += len(filt_points)
        return total_points

    def _get_buckets_len(
        self, poros_width: float, buckets_starts: list[Decimal],
        target_rings: list[int]
    ) -> tuple[dict[Decimal, int], dict[Decimal, list[tuple[int, int]]]]:
        """
        Counts how many points are within each bucket and which
        (ring, well_id) pairs make the buckets
        
        args: 
            poros_width (float): The buckets porosity width
            buckets_starts (list[float]): The porosity start 
            for every bucket
            target_rings (list[int]): The rings to count points 

        return:
            A tuple (buckets_len, buckets_origin) where 
            buckets_len is a dict with bucket porosity start as keys and num of
            items in it as values, buckets_origin is a dict with bucket porosity
            start as keys and a list of (ring, well_id) pairs as values 
        """
        buckets_len = defaultdict(int)
        buckets_origin = defaultdict(list)
        for ring in target_rings:
            for well_id in self._wells_id_list:
                chunk_data = self._get_values_hook(ring, well_id)
                for p in buckets_starts:
                    points_within = sum((chunk_data['phi'] >= p)
                                        & (chunk_data['phi'] < p + poros_width))
                    buckets_len[p] += points_within
                    if points_within > 0:
                        buckets_origin[p].append((ring, well_id))
        return buckets_len, buckets_origin

    def _perf_sampling(self, it):
        if self._sampler == 'v2':
            self._perf_sampling_v2(it)
            return

        # Sample each ring individually
        # TODO: Sampling is memory inefficient: all data from a given ring is
        # first compiled and then sampled. Maybe later change the sampler to
        # receive as input the points from a ring/well pair.

        total_n_points = len(self)

        for ring_being_sampled in self._rings_list:
            # Compile all points from a ring
            ring_points = []
            for well_id in self._wells_id_list:
                ring_points.extend(
                    self._get_values_hook(ring_being_sampled, well_id))

            assert_msg = f"[TrialDataBase][perf_sampling][it{it}] "\
                         f"Ring {ring_being_sampled} points is empty!"
            assert len(ring_points) > 0, assert_msg

            # Perform sampling
            #Tem que usar o Config.ring_range_to_expand e passar o start_ring
            # no lugar do it. O ring está certo
            first_ring_to_prop_this_it, _ = self._config.ring_range_to_expand(
                it)
            new_ring_points = self._sampler.sample(np.array(ring_points),
                                                   total_n_points,
                                                   first_ring_to_prop_this_it,
                                                   ring_being_sampled)

            assert_msg = f"[TrialDataBase][perf_sampling][it{it}] New "\
                         f"ring points is empty!"
            assert new_ring_points.size > 0, assert_msg

            # Split all points by well_id and add them to a dict
            new_rings_dict = dict()
            field_names = [i for i, j in self._base_data_type]

            for well_id in self._wells_id_list:
                new_rings_dict[well_id] = new_ring_points[
                    new_ring_points['well_id'] == well_id][field_names]

            # Update the internal concrete data with the sampled points
            # Existing points are deleted
            self._set_ring_hook(ring_being_sampled, new_rings_dict, True)

            # # Used for getting the sampled coords for
            # # sampling integration testing.
            # print(f'------------------- ring{ring_key}:')
            # print(new_ring[['x', 'y', 'z']])

    def _should_commit_feature_hook(self):
        '''
        Hook used by concurrent implementations of TrialDataBase.
        Default behavior is: return True, i.e., all processes perform
        update_feature().

        Override implementations should lock while committing is being
        performed, and return False afterwards. This keeps the profiling 
        times consistent. I.e., the commit time of a non-committing process
        will not be accounted on get_values().
        '''
        return True

    def _done_commit_feature_hook(self):
        '''
        Hook used by concurrent implementations of TrialDataBase.
        Default behavior is: do nothing, since all processes perform 
        update_feature() there is no need for syncing.

        Override implementations should release all remaining processes
        locked for feature committing.
        '''
        pass
