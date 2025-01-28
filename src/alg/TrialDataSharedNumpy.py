from multiprocessing import shared_memory, resource_tracker
import numpy as np
import mpi4py

# For some unknown buggy reason using the import below results in pytest not
# executing the Popen('mpirun...') commands. It just skips the execution...
# Still, just importing mpi4py seems ok..... nice...
# from mpi4py import MPI

from TrialDataSharedBase import TrialDataSharedBase


class TrialDataSharedNumpy(TrialDataSharedBase):
    '''
    Shared memory implementation through numpy.

    '''

    def __init__(self,
                 target_wells_list,
                 porosity_data,
                 config,
                 should_consider_sampling: bool = True):
        # Currently no initialization is needed
        super(TrialDataSharedNumpy,
              self).__init__(target_wells_list, porosity_data, config,
                             should_consider_sampling)

        # Representation of data storage index. Actual concrete data used
        # numpy objects to be stores within the dict's, by ring,well.
        # Two objects are created, one for shared data access, and one for
        # local data (i.e., current feature data).
        self._data_shr = dict()
        self._data_local = dict()

        # Dict of (ring, well) pairs is used to allow deleting a single
        # shared memory region for sampling.
        self._shm_objects = dict()

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

    def __del__(self):
        self._del_all_concrete()

    def _get_shd(self, ring, well, chunk_slice=None):
        '''
        Returns the concrete numpy reference to the shared data structure.
        Returns an empty np array if the ring,well pair is not present.
        '''
        # Check if there is a ring r.
        shm_ring_dict = self._data_shr.get(ring)
        if shm_ring_dict is None:
            return np.empty(0)

        # Retrieve all data
        shm_well_data = shm_ring_dict.get(well, np.empty(0))

        if chunk_slice is not None:
            if len(shm_well_data) > 0:
                return shm_well_data[chunk_slice]
            else:
                return np.empty(0)
        else:
            return shm_well_data

    def _get_local(self, ring, well, chunk_slice=None):
        '''
        Returns a concrete reference to the local data structure.
        This concrete structure have numpy index semantics.
        Returns an empty np array if the ring,well pair is not present.
        '''

        if chunk_slice is not None:
            return self._data_local.get(ring, {}).get(well, np.empty(0))[chunk_slice]
        else:
            return self._data_local.get(ring, {}).get(well, np.empty(0))

    def _update_shd_col(self, ring, well, cols, data):
        '''
        Updates a single column on the shared data structure.
        '''
        self._data_shr[ring][well][cols] = data

    def _update_local_col(self, ring, well, data):
        '''
        Updates the last column on the local data structure.
        '''
        self._data_local[ring][well][:] = data

    def _alloc_empty_ring_well_last_feature_concrete(self, length, ring, well):
        # Only the space for a single column is allocated.
        self._data_local[ring][well] = np.zeros((length), dtype=np.float64)

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
            shm_object = shared_memory.SharedMemory(name=shm_name, create=False)
            # This unregister deals with an obnoxious warning from
            # resource_tracker, which senses leaking shm objects.
            # All shm objects are properly cleaned on __del__().
            resource_tracker.unregister(shm_object._name, 'shared_memory')

        self._shm_objects[(ring, well)] = shm_object

        # Store the array data which wraps a shared memory region
        self._data_shr[ring][well] = np.ndarray((length),
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

    def _del_single_ring_well(self, ring, well):
        shm = self._shm_objects.pop((ring, well))
        shm.close()
        if self._mpi_local_comm is None or self._is_resp_rank:
            shm.unlink()

    def _del_all_concrete(self):
        '''
        Clears all data managed by the concrete class.
        All processes should call this method to avoid being locked at 
        the barrier.
        This implementation is idempotent.
        '''
        for shm_object in self._shm_objects.values():
            shm_object.close()

        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()

        if self._mpi_local_comm is None or self._is_resp_rank:
            for shm_object in self._shm_objects.values():
                shm_object.unlink()
            self._shm_objects = dict()

        # Clear shm objects list, otherwise other calls to _del_all_concrete
        # may attempt to unlink already unlinked shm objects
        self._shm_objects = dict()

        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()
