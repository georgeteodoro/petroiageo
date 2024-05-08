import numpy as np
from mpi4py import MPI
from multiprocessing import shared_memory, resource_tracker

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
        super(TrialDataSharedNumpy, self).__init__(
            target_wells_list, porosity_data, config, should_consider_sampling)

        # Shared memory objects used to get data for numpy objects
        # (self._data_shr). This is necessary for cleanup on __del__().
        self._shm_objects = []

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
                [np.int64(mpi_local_rank), MPI.LONG], [rank_list, MPI.LONG])
            self._resp_rank = rank_list.min()
            self._is_resp_rank = mpi_local_rank == self._resp_rank
        else:
            print("[TrialDataSharedBase] _mpi_local_comm is None. "\
                  "Ignore if unittesting.")

    def __del__(self):
        for shm_object in self._shm_objects:
            shm_object.close()
        
        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()
        
        if self._is_resp_rank:
            for shm_object in self._shm_objects:
                shm_object.unlink()
        
        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()

    def _alloc_empty_ring_well_last_feature_concrete(self,
                                                     length,
                                                     last_feature=False):
        # Only the space for a single column is allocated.
        return np.zeros((length), dtype=np.float64)

    def _alloc_empty_ring_well_concrete(self, length, last_feature=False):
        '''
        Allocate an empty concrete object to store shared data for a
        ring/well pair, or the single column for the current feature.
        '''

        # Only a single responsible rank allocates the shared memory region
        if self._is_resp_rank:
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
        return np.ndarray((length),
                          dtype=self._cur_data_type,
                          buffer=shm_object.buf)
