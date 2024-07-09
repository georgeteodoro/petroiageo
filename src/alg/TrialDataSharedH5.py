import numpy as np
import mpi4py
import os
import h5py

from TrialDataSharedBase import TrialDataSharedBase


class TrialDataSharedH5(TrialDataSharedBase):
    '''
    Shared memory implementation through H5.
    '''

    def __init__(self,
                 target_wells_list,
                 porosity_data,
                 config,
                 should_consider_sampling: bool = True):
        # Currently no initialization is needed
        super(TrialDataSharedH5,
              self).__init__(target_wells_list, porosity_data, config,
                             should_consider_sampling)

        # These default values are required for unittesting since without
        # mpi_local there cannot be a responsible process
        resp_rank = 0
        mpi_local_rank = 0
        self._is_resp_rank = True

        self._mpi_local_comm = config.get_param('mpi_local_comm')
        if self._mpi_local_comm is not None:
            # Local node processes communicator. Required to coordinate the
            # creation of a single shared memory region for all processes.
            mpi_local_rank = self._mpi_local_comm.Get_rank()

            # Set a single intra-node process as the creator of the
            # shared-memory LRU list. This is the responsible rank.
            rank_list = np.zeros(self._mpi_local_comm.Get_size(),
                                 dtype=np.int64)
            self._mpi_local_comm.Allgather(
                [np.int64(mpi_local_rank), mpi4py.MPI.LONG],
                [rank_list, mpi4py.MPI.LONG])
            resp_rank = rank_list.min()
            self._is_resp_rank = mpi_local_rank == resp_rank
        else:
            print("[TrialDataSharedBase] _mpi_local_comm is None. "\
                  "Ignore if unittesting.")

        # Define H5 filenames for shared and local data
        self._shd_filename = f"/tmp/TD-shr-r{resp_rank}-tmp.h5"
        self._local_filename = f"/tmp/TD-local-r{mpi_local_rank}-tmp.h5"

        # If the cur file exists, it should be deleted
        if self._is_resp_rank and os.path.exists(self._shd_filename):
            os.remove(self._shd_filename)
        if os.path.exists(self._local_filename):
            os.remove(self._local_filename)

        # H5 MPI config for multiple processes opening the same file
        if self._mpi_local_comm is not None:
            self._mpi_kwargs = {
                "driver": "mpio",
                "comm": config.get_param("mpi_local_comm"),
            }
        else:
            # Empty config for unittseting, when not mpi comm is used
            self._mpi_kwargs = {}

        # File creating is a collective operation, thus must be performed
        # by all processes
        self._shd_h5 = h5py.File(f'{self._shd_filename}', 'a',
                                 **self._mpi_kwargs)

        # Create local H5 file
        self._local_h5 = h5py.File(f'{self._local_filename}', 'w')

    def __del__(self):
        self._del_all_concrete()

    def _get_shd(self, ring, well, chunk_slice=None):
        '''
        Returns a concrete numpy array, extracted from the h5 file.
        Returns an empty np array if the ring,well pair is not present.
        '''
        # Get dset for the ring,well pair
        dset_name = f'r{ring}-w{well}'
        dset = self._shd_h5.get(dset_name)

        if dset is None:
            return np.empty(0)
        else:
            if chunk_slice is None:
                return dset[:]
            else:
                return dset[chunk_slice]

    def _get_local(self, ring, well, chunk_slice=None):
        '''
        Returns a concrete reference to the local data structure.
        This concrete structure have numpy index semantics.
        Returns an empty np array if the ring,well pair is not present.
        '''

        # Get dset for the ring,well pair
        dset_name = f'r{ring}-w{well}'
        dset = self._local_h5.get(dset_name)

        if dset is None:
            return np.empty(0)
        else:
            if chunk_slice is None:
                return dset[:]
            else:
                return dset[chunk_slice]

    def _update_shd_col(self, ring, well, cols, data):
        '''
        Updates a single column on the shared data structure.
        '''

        # Get dset for the ring,well pair
        dset_name = f'r{ring}-w{well}'
        dset = self._shd_h5.get(dset_name)
        assert dset is not None

        if len(cols) > 1:
            for i, col in enumerate(cols):
                dset[col] = [item[i] for item in data]
        else:
            dset[cols[0]] = data

    def _update_local_col(self, ring, well, data):
        '''
        Updates the last column on the local data structure.
        '''

        # Get dset for the ring,well pair
        dset_name = f'r{ring}-w{well}'
        dset = self._local_h5.get(dset_name)
        assert dset is not None

        dset[:] = data[:]

    def _alloc_empty_ring_well_last_feature_concrete(self, length, ring, well):
        dset_name = f'r{ring}-w{well}'

        # If performing sampling, there should already be a dataset with this
        # name, thus we should delete the old data first
        if self._local_h5.get(dset_name) is not None:
            del self._local_h5[dset_name]

        # Only the space for a single column is allocated.
        self._local_h5.create_dataset(dset_name, (length, ), dtype=np.float64)

    def _alloc_empty_ring_well_concrete(self, length, ring, well):
        '''
        Allocate an empty concrete object to store shared data for a
        ring/well pair, or the single column for the current feature.
        '''

        dset_name = f'r{ring}-w{well}'

        # If performing sampling, there should already be a dataset with this
        # name, thus we should delete the old data first
        if self._shd_h5.get(dset_name) is not None:
            del self._shd_h5[dset_name]

            # Sync is required to make sure no other process begins creating
            # the new dataset before all processes delete the old one before.
            if self._mpi_local_comm is not None:
                self._mpi_local_comm.Barrier()

        # h5py.create_detaset is a collective operation, thus must be
        # performed by all processes
        self._shd_h5.create_dataset(dset_name, (length, ),
                                    dtype=self._cur_data_type)

    def _new_ring_hook(self, ring):
        '''
        Creation of datasets is done by ring,well pair, not just ring.
        Does nothing.
        '''

        pass

    def _clear_trial_data_hook(self):
        self._del_all_concrete()

    def _del_all_concrete(self):
        '''
        Delete all data by deleting the H5 file. The local file is deleted 
        individually by each owing process, while shared data is deleted
        only by the responsible process.
        '''

        if os.path.exists(self._local_filename):
            os.remove(self._local_filename)

        # Create new local H5 file
        self._local_h5 = h5py.File(f'{self._local_filename}', 'w')

        # Responsible rank should delete shared data after waiting sync of
        # all remaining processes. This solves the race condition of
        # deleting a h5 file when another process might still be
        # accessing it.
        if self._is_resp_rank:
            if self._mpi_local_comm is not None:
                self._mpi_local_comm.Barrier()
            if os.path.exists(self._shd_filename):
                os.remove(self._shd_filename)
        else:
            self._mpi_local_comm.Barrier()

        # File creating is a collective operation, thus must be performed
        # by all processes
        self._shd_h5 = h5py.File(f'{self._shd_filename}', 'a',
                                 **self._mpi_kwargs)