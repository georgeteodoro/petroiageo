from multiprocessing import shared_memory, resource_tracker
import numpy as np
import mpi4py

# For some unknown buggy reason using the import below results in pytest not
# executing the Popen('mpirun...') commands. It just skips the execution...
# Still, just importing mpi4py seems ok..... nice...
# from mpi4py import MPI

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
            self._resp_rank = rank_list.min()
            self._is_resp_rank = mpi_local_rank == self._resp_rank
        else:
            print("[TrialDataSharedBase] _mpi_local_comm is None. "\
                  "Ignore if unittesting.")

        # Define H5 filenames for shared and local data
        self._shd_filename = f"/tmp/TD-shr-tmp.h5"
        self._local_filename = f"/tmp/TD-r{mpi_local_rank}-tmp.h5"

        # If the cur file exists, it should be deleted
        if os.path.exists(self._shd_filename):
            os.remove(self._shd_filename)
        if os.path.exists(self._local_filename):
            os.remove(self._local_filename)

        # Responsible rank creates the shared H5 file, 
        # while remaining ranks open it
        if self._resp_rank:
            self._shd_h5 = h5py.File(f'{self._shd_filename}', 'w')
            if self._mpi_local_comm is not None:
                self._mpi_local_comm.Barrier()
        else:
            self._mpi_local_comm.Barrier()
            self._shd_h5 = h5py.File(f'{self._shd_filename}', 'r+')

        # Create local H5 file
        self._local_h5 = h5py.File(f'{self._local_filename}', 'w')
        # self._last_col_name = 'l'

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
        dset_name = f'r{ring}w{well}'
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

        for i, col in enumerate(cols):
            dset[field] = [item[i] for item in data]

        # self._data_shr[ring][well][cols] = data

    def _update_local_col(self, ring, well, data):
        '''
        Updates the last column on the local data structure.
        '''

        # Get dset for the ring,well pair
        dset_name = f'r{ring}-w{well}'
        dset = self._local_h5.get(dset_name)
        assert dset is not None

        dset[:] = data[:]

        # self._data_local[ring][well][:] = data


    def _alloc_empty_ring_well_last_feature_concrete(self,
                                                     length, ring, well):
        # Only the space for a single column is allocated.
        # self._data_local[ring][well] = np.zeros((length), dtype=np.float64)
        dset_name = f'r{ring}-w{well}'
        self._local_h5.create_dataset(dset_name,
                                      (length, ),
                                      dtype=np.float64)


# ========================================================
    def _alloc_empty_ring_well_concrete(self, length, ring, well):
        '''
        Allocate an empty concrete object to store shared data for a
        ring/well pair, or the single column for the current feature.
        '''

        # On the case of an empty ring,well pair, just create an empty
        # numpy array with the correct shape/dtype.
        if length == 0:
            return np.zeros((0), dtype=self._cur_data_type)

        # Only a single responsible rank allocates the shared memory region
        if self._mpi_local_comm is None or self._is_resp_rank:
            shm_object = shared_memory.SharedMemory(
                create=True, size=(length * self._cur_data_type.itemsize))

            # Broadcasts the shared memory name to other processes
            # on the same node
            if self._mpi_local_comm is not None:
                self._mpi_local_comm.bcast(shm_object.name,
                                           root=self._resp_rank)
            else:
                print("[TrialDataSharedBase] _mpi_local_comm is None. "\
                      "Ignore if unittesting.")

        else:
            # Receive the shared memory name for the allocating process
            shm_name = self._mpi_local_comm.bcast(None, root=self._resp_rank)

            # Opens the shared memory region, previously created by
            # the allocating process
            shm_object = shared_memory.SharedMemory(name=shm_name,
                                                    create=False)
            # This unregister deals with an obnoxious warning from
            # resource_tracker, which senses leaking shm objects.
            # All shm objects are properly cleaned on __del__().
            resource_tracker.unregister(shm_object._name, 'shared_memory')

        self._shm_objects.append(shm_object)

        # Return array data which wraps a shared memory region
        self._data_shr[ring][well]= np.ndarray((length),
                          dtype=self._cur_data_type,
                          buffer=shm_object.buf)

    def _new_ring_hook(self, ring):
        '''
        A new ring is a dict of data by well_id
        '''
        self._data_shr[ring] = dict()
        self._data_local[ring] = dict()

    def _clear_trial_data_hook(self):
        '''
        Delete all data, resetting internal data to an empty dict.
        '''

        # Local data can be deleted individually
        del self._data_local
        self._data_local = dict()

        # Delete all concrete data stored. Coordination, i.e. which process
        # actually deletes shared data, is solved by the concrete
        # implementation.
        self._del_all_concrete()

    def _del_all_concrete(self):
        '''
        Clears all data managed by the concrete class.
        All processes should call this method to avoid being locked at 
        the barrier.
        '''
        for shm_object in self._shm_objects:
            shm_object.close()

        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()

        if self._mpi_local_comm is None or self._is_resp_rank:
            for shm_object in self._shm_objects:
                shm_object.unlink()

        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()
