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

        # Check if this is the updating process
        self._is_UP = True
        self._mpi_local_comm = config.get_param('mpi_local_comm')

        # Internal propagation state
        self._current_ring = 0
        # self._current_feature_id = -1
        # self._current_features = []

        # Porosity HDF5 dataset
        self._porosity_data = porosity_data

        self._config = config
        self._wells_id_list = target_wells_ids_list

        

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

        # Prepare shared data
        if self._is_UP:
            # Create a temporary p-block lists dict. This could also be
            # implemented in file+mmap, but this should suffice...
            # For each p-block there is a list of ndarrays to be 
            # concatenated at the end.
            base_type = list(self._porosity_data.dtype.fields.items())
            base_type.pop(-2) # remove ring field
            base_type = np.dtype([(x,y) for x,(y,z) in base_type])
            chunk_len = prod(self._porosity_data.chunks)

            # p_block_tmp_arrays = defaultdict(
            #     lambda: np.ndarray(chunk_len, base_type))
            # p_block_tmp_idxs = defaultdict(lambda: 0)
            p_block_lists = defaultdict(lambda: [])

            cs = [ceil(s/c) for (s,c) in zip(
                self._porosity_data.shape, self._porosity_data.chunks)]
            cs = prod(cs)

            # Scan each porosity chunk
            for p_chunk_slice in tqdm(self._porosity_data.iter_chunks(), total=cs):

                # Skip this chunk if there cannot be any points withing it
                if not common.has_points_within_chunks(target_wells_coords, 
                        self._current_ring, prep_it, p_chunk_slice):
                    continue

                # Load porosity data chunk
                chunk_np = self._porosity_data[p_chunk_slice]

                # Flatten porosity data shape
                chunk_np = chunk_np.reshape(-1)

                # Assign each point to the correct p-block 
                # (insertion-search-like)
                for p in chunk_np:
                    (x,y,z,phi,real,ring,well_id) = p
                    # If slow, use an array with an incrementing ID instead
                    # of a list. If still slow, maybe numba? 
                    # NOPE, arrays are slower given the allocation cost

                    # idx = p_block_tmp_idxs[ring,well_id]
                    # p_block_tmp_arrays[ring,well_id][idx] = (
                    #     x,y,z,phi,real,well_id)
                    if ring >= self._current_ring and ring <= prep_it:
                        p_block_lists[ring,well_id].append(
                            (x,y,z,phi,real,well_id))

            # # Perform sampling (it's in memory anyway...)
            # p_block_lists = sampling(p_block_lists)

            # Create the shared mmap-ing of each p-block with 
            # an underlying file
            for (ring,well_id), points in p_block_lists.items():
                # Read points as an np.array
                # points_np = np.array(points, dtype=base_type)
                # print(points_np.shape)
                # print(points_np.dtype)
                # print(points[0])
                # print(points_np[0])
                # 0/0
                data_size = base_type.itemsize * len(points)

                filename = f'/tmp/TDF-r{ring}-w{well_id}.dat'
                file = open(filename, 'w+')
                file.truncate(data_size)
                # fd = os.open(filename, 
                #     flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
                # os.ftruncate(fd, sys.getsizeof(points_np))
                # mmap_buffer = mmap.mmap(fd,
                mmap_buffer = mmap.mmap(file.fileno(), data_size,
                        flags=mmap.MADV_SEQUENTIAL | mmap.MAP_SHARED)
                        # prot=mmap.PROT_READ)

                np_array = np.ndarray(len(points), 
                    base_type, buffer=mmap_buffer)
                
                np_array[:] = points
                print(points[0])
                print(np_array[0])
                print(points[-1])
                print(np_array[-1])
                mmap_buffer.close()
                file.close()
                0/0
            0/0


        # Done preparing mmap's
        self._mpi_local_comm.Barrier()
        
        # Open p-blocks mmap's with read-only

        # Close read-write mmap's
        if self._is_UP:
            close()

        self._mpi_local_comm.Barrier()

        # Prepare local data

        self._current_ring = prep_it




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
