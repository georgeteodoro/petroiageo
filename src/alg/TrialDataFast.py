from abc import ABC, abstractmethod
from h5py import Dataset, h5d
import numpy as np
from numpy.lib import recfunctions as rfn
from time import time, sleep
from math import ceil, prod
from collections import defaultdict
from tqdm import tqdm
import sys
import os
import mmap

import common
from config_parser import Config
from data_filter import WellsSingleRingDataFilter
from sampler import ChunkSamplerV1


class TrialDataFast:
    '''
    Fast, and possibly final, implementation of TrialData.
    Data is stored in well,ring pair blocks (p-block).
    Each p-block is maintained in a memory-mapped tmp file.
    Only a single process per node is allowed to write on
    p-blocks. However, all processes have a read-only file 
    descriptor for all p-blocks. When updating data, the 
    updating process opens a new write-enabled file descriptor,
    which is closed after use. 

    The current trial column is maintained individually by 
    each process.

    Sampling is only performed after all p-blocks are filled.
    '''

    def __init__(self,
                 target_wells_ids_list,
                 porosity_data: Dataset,
                 config: Config,
                 should_consider_sampling: bool = True):

        self._config = config
        self._n_features = config.alg['max_num_features']
        self._wells_id_list = target_wells_ids_list

        # Set the datatype for points
        self._base_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('real', np.int64),
            ('well_id', np.int64),
        ]
        self._base_data_type_names = [n for (n, t) in self._base_data_type]
        self._cur_data_type = self._base_data_type + [
            (f'f{f}', np.float64) for f in range(self._n_features)
        ]
        self._cur_data_type = np.dtype(self._cur_data_type)

        # Check if this is the updating process
        self._is_UP = True
        self._mpi_local_comm = config.get_param('mpi_local_comm')

        # Internal propagation state
        self._current_ring = 0
        # self._current_feature_id = -1
        # self._current_features = []

        # Porosity HDF5 dataset
        self._porosity_data = porosity_data

        self._all_ring_well_pairs = set()
        self._rings_list = []
        self._wells_list = []

        # === Shared data variables ================================
        self._mmap_file_prefix = 'TDF-'
        self._mmap_file_location = '/tmp'

        # Dict of shared read-only p-blocks.
        # Key:   (ring,well)
        # Value: np.ndarray
        self._shared_p_blocks = dict()

        # List of shared read-only p-blocks pointers.
        # Value: (FD, mmap)
        self._shared_ctrl_p_blocks = []

        # === Local data variables ================================
        self._local_p_blocks = dict()

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
                 prepare. For running iteration 20 (will propagate ring 20
                 after at the end of iteration 20) prep_it should be 19.
                 I.e., rings 0-19 are loaded.
        '''

        assert prep_it>=0, f"[TrialDataBase][prepare_porosity] "\
            f"First iteration is 0, but received current iteration {prep_it}."

        profile = self._config.get_param('prof_trial_prep_porosity')

        target_wells_coords = self._config.get_coords_of_target_wells_ids(
            self._wells_id_list)

        # Set the first feature, even if it's empty
        self._current_feature_id = 0
        self._current_features = []
        self._current_features.append(f'f{self._current_feature_id}')

        # Prepare shared data
        if self._is_UP:
            # Create a temporary p-block lists dict. This could also be
            # implemented in file+mmap, but this should suffice...
            # For each p-block there is a list of ndarrays to be
            # concatenated at the end.

            # Testing np arrays. Sadly, it is slower...
            # p_block_tmp_arrays = defaultdict(
            #     lambda: np.ndarray(chunk_len, base_type))
            # p_block_tmp_idxs = defaultdict(lambda: 0)

            # Temporary in-memory lists of all p-blocks
            p_block_lists = defaultdict(lambda: [])

            # TODO: If loading all to memory is too expensive we can do two
            # passes: (1) to get sizes of each p-block and (2) to write
            # directly to the file-backed p-block mmap.
            # p_block_sizes = defaultdict(lambda: 0)

            cs = [
                ceil(s / c) for (s, c) in zip(self._porosity_data.shape,
                                              self._porosity_data.chunks)
            ]
            cs = prod(cs)

            # Scan each porosity chunk
            for p_chunk_slice in tqdm(self._porosity_data.iter_chunks(),
                                      total=cs):

                # Skip this chunk if there cannot be any points withing it
                if not common.has_points_within_chunks(
                        target_wells_coords, self._current_ring, prep_it,
                        p_chunk_slice):
                    continue

                # Load porosity data chunk
                chunk_np = self._porosity_data[p_chunk_slice]

                # Flatten porosity data shape
                chunk_np = chunk_np.reshape(-1)

                # Assign each point to the correct p-block
                # (insertion-search-like)
                for p in chunk_np:
                    (x, y, z, phi, real, ring, well_id) = p
                    # If slow, use an array with an incrementing ID instead
                    # of a list. If still slow, maybe numba?
                    # NOPE, arrays are slower given the allocation cost

                    # idx = p_block_tmp_idxs[ring,well_id]
                    # p_block_tmp_arrays[ring,well_id][idx] = (
                    #     x,y,z,phi,real,well_id)
                    if ring >= self._current_ring and ring <= prep_it:
                        p_block_lists[ring, well_id].append(
                            (x, y, z, phi, real, well_id))
                        # p_block_sizes[ring,well_id] += 1

            # TODO sampling....
            # # Perform sampling (it's in memory anyway...)
            # p_block_lists = sampling(p_block_lists)

            # Create the shared mmap-ing of each p-block with
            # an underlying file
            for (ring, well_id), points in tqdm(p_block_lists.items()):
                data_size = self._cur_data_type.itemsize * len(points)

                self._all_ring_well_pairs.add((ring, well_id))
                self._rings_list.append(ring)
                self._wells_list.append(well_id)

                # Create tmp file for file backing
                filename = (f'{self._mmap_file_location}/'
                            f'{self._mmap_file_prefix}r{ring}-w{well_id}.dat')
                file = open(filename, 'w+')
                file.truncate(data_size)
                # fd = os.open(filename,
                #     flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                # os.ftruncate(fd, sys.getsizeof(points_np))
                # mmap_buffer = mmap.mmap(fd,
                mmap_buffer = mmap.mmap(file.fileno(),
                                        data_size,
                                        flags=mmap.MADV_SEQUENTIAL
                                        | mmap.MAP_SHARED)
                # prot=mmap.PROT_READ)

                np_array = np.ndarray(len(points),
                                      self._cur_data_type,
                                      buffer=mmap_buffer)

                # Copy data to mmap
                np_array[self._base_data_type_names] = points

                self._shared_p_blocks[ring, well_id] = np_array
                self._shared_ctrl_p_blocks.append((file, mmap_buffer))

                # print(np_array[0])

        # Done preparing mmap's
        self._mpi_local_comm.Barrier()

        # Get a list of all p-blocks created
        all_p_blocks_fnames = filter(lambda s: self._mmap_file_prefix in s,
                                     os.listdir(self._mmap_file_location))

        # Open p-blocks mmap's with read-only for non-updating-processes
        if not self._is_UP:
            for p_block_fname in all_p_blocks_fnames:
                # Find ring,well
                (r, w) = p_block_fname.replace(self._mmap_file_prefix,
                                               '').replace('.dat', '').replace(
                                                   'r',
                                                   '').replace('w',
                                                               '').split('-')
                ring_well_pair = (int(r), int(w))

                self._all_ring_well_pairs.add(ring_well_pair)
                self._rings_list.append(r)
                self._wells_list.append(w)

                file = open(f'{self._mmap_file_location}/{p_block_fname}', 'r')
                mmap_buffer = mmap.mmap(
                    file.fileno(),
                    0,  # full size
                    flags=mmap.MADV_SEQUENTIAL
                    | mmap.MAP_SHARED,
                    prot=mmap.PROT_READ)

                p_block_len = mmap_buffer.size(
                ) // self._cur_data_type.itemsize
                np_array = np.ndarray(p_block_len,
                                      self._cur_data_type,
                                      buffer=mmap_buffer)

                self._shared_p_blocks[ring_well_pair] = np_array
                self._shared_ctrl_p_blocks.append((file, mmap_buffer))

        # All p-blocks were opened for all processes
        self._mpi_local_comm.Barrier()

        # Prepare local data
        for ring_well_pair, np_array in self._shared_p_blocks.items():
            self._local_p_blocks[ring_well_pair] = np.zeros((len(np_array), ),
                                                            dtype=np.float64)

        self._current_ring = prep_it

    def commit_feature(self, feature, disp):
        '''
        Commits the current feature then set up the next feature.
        '''
        assert self._current_feature_id >= 0, "[TrialDataBase][commit_feature] "\
            "Committing feature before prepare_porosity."
        assert self._current_feature_id < self._n_features, \
            "[TrialDataBase][commit_feature] Committing beyond last feature: "\
            f"cur_feature={self._current_feature_id} "\
            f"n_features={self._n_features}."

        # Only a single process per node should commit to shared mmap
        if self._is_UP:
            self.update_feature(feature, disp, update_shared=True)

        # Sync all local processes
        self._mpi_local_comm.Barrier()

        self._current_feature_id += 1
        self._current_features.append(f'f{self._current_feature_id}')

    def update_feature(self, feature, disp, update_shared=False):
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

        for ring_well_pair in self._all_ring_well_pairs:
            t11 = time()

            # Check if there are points for a ring/well pair. Example for
            # when there are no points: well w has already propagated
            # all it could, being surrounded by other wells, thus there
            # may be a ring for which it has no points.
            cur_values = self._shared_p_blocks[ring_well_pair]
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

            if update_shared:
                feature_str = f'f{self._current_feature_id}'
                self._shared_p_blocks[ring_well_pair][
                    feature_str] = filtered_feature_data
            else:
                # Assign feature data to the last col (i.e., current col
                # being updated)
                self._local_p_blocks[ring_well_pair] = filtered_feature_data
            t15 = time()

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
            length += len(self._local_p_blocks[ring, w])
        return length

    def __len__(self):
        length = 0
        for r in self._rings_list:
            length += self._ring_size(r)
        return length

    def _get_values(self, wells_to_retrieve, chunk_id):
        '''
        Helper function for filtering trial_data.
        Data is retrieved by ring and well_id until a chunk is reached.
        '''

        n_training_chunks = self._config.get_param('n_training_chunks')
        profile = self._config.get_param('prof_TD_get_values')

        prep_slice_time = 0
        get_val_hook_time = 0
        to_list_time = 0
        append_time = 0

        # Output collection of data. Each data chunk (ring/well pair) is
        # appended to the lists bellow. Later these are concatenated, avoiding
        # using extend and reallocating data.
        X = []
        y = []

        def _update_X_Y(new_points_shared, new_points_local, cur_features,
                        is_val, X, Y):
            # If empty points, don't bother
            if new_points_local.size == 0:
                return

            # Only real points should be used for validation
            if is_val:
                real_cond = new_points_shared['real'] == common.RealValues.real
                new_points_shared = new_points_shared[real_cond]
                new_points_local = new_points_local[real_cond]

                # There can be no
                if new_points_local.size == 0:
                    return

            # Split X from y
            new_points_shared_X = new_points_shared[cur_features[:-1]]

            # Must reshape to a 2D array since this is the only format that
            # lgb accepts.
            new_points_local_X = new_points_local.reshape(
                (len(new_points_local), 1))

            new_points_y = new_points_shared['phi']

            # 1 feature means no shared data has been committed.
            # Thus, new_points_shared_X should be empty.
            if len(cur_features) == 1:
                new_points_X = new_points_local_X
            else:
                # Remove structured array info with zero-copy.
                new_points_shared_X = rfn.structured_to_unstructured(
                    new_points_shared_X)
                new_points_X = np.concatenate(
                    (new_points_shared_X, new_points_local_X), axis=1)

            # t3 = time()
            # to_list_time += t3 - t2

            # Add them to output arrays
            X.append(new_points_X)
            y.append(new_points_y)

            # t4 = time()
            # append_time += t4 - t3

        # Fill training data, one ring at a time, one well at a time
        for ring_well_pair in self._all_ring_well_pairs:
            t0 = time()
            rw_len = len(self._local_p_blocks[ring_well_pair])
            is_val = len(wells_to_retrieve) == 1

            # Since validation only uses real data, and real data is
            # only present in ring 0, validation and ring>0 can be skipped
            r, w = ring_well_pair
            if is_val and r > 0:
                continue

            # If chunking is used (i.e., not validation or test data)
            if chunk_id >= 0:
                # Calculate how many points from a ring/well_id pair
                # this chunk should have
                points_per_rw = int(ceil(rw_len / n_training_chunks))

                # Generate a chunk slice for the ring/well pair
                beg = chunk_id * points_per_rw
                end = (chunk_id + 1) * points_per_rw
                end = min(end, rw_len)
                cur_slice = slice(int(beg), int(end))
            else:
                cur_slice = slice(0, rw_len)
            t1 = time()
            prep_slice_time += t1 - t0

            # Retrieve current chunk slice from the backend storage
            new_points_shared = self._shared_p_blocks[ring_well_pair][
                cur_slice]
            new_points_local = self._local_p_blocks[ring_well_pair][cur_slice]

            t2 = time()
            get_val_hook_time += t2 - t1

            # Update X and Y with the new points to be returned for
            # the current chunk_id
            _update_X_Y(new_points_shared, new_points_local,
                        self._current_features, is_val, X, y)

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
