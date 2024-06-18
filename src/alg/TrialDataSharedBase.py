from abc import ABC, abstractmethod
import fasteners  # inter-process, intra-node lock
import numpy as np

from TrialDataBase import TrialDataBase


class TrialDataSharedBase(TrialDataBase, ABC):
    '''
    Subclass of TrialData with shared storage within the same node.
    This strategy works by sharing the committed features data, while
    maintaining the current feature data individually for each process.
    Since this class uses shared-memory, it can be instantiated multiple
    times, however, for each instance there will be only 1 column worth of 
    data allocated, with the remaining 10 columns residing in shared-memory
    space.

    Initialization is thread-safe and concurrent. All processes try to get
    a file lock for shared-memory initialization, if the shared-memory space
    was not yet initialized. On failing to get the lock, the it first check 
    for the shared-memory space existence. Thus, only the first process to 
    acquire the file lock actually initializes the shared-memory space.

    When retrieving data, the same interfaces are preserved. It is the
    responsibility of this class to fetch both shared-memory and current
    feature data and compile it for return.

    This is an abstract class, abstracting the backend for storing data. 
    Both feature data and shared data are abstracted.
    '''

    def __init__(self,
                 target_wells_list,
                 porosity_data,
                 config,
                 should_consider_sampling: bool = True):
        # Currently no initialization is needed
        super(TrialDataSharedBase,
              self).__init__(target_wells_list, porosity_data, config,
                             should_consider_sampling)

        # Load config
        self._mpi_local_comm = config.get_param('mpi_local_comm')

        # Inter-process lock for updating the shared-memory data. This can be
        # for deleting data (_clear_trial_data_hook()) or committing data.
        # This lock is also used for creating shared-memory data regions for
        # (ring, well_id) pairs.
        self._shm_lock = fasteners.InterProcessLock(
            '/tmp/TrialDataSharedBase.lock')

        self._is_last_col_empty = True
        self._commiting_feature = False

    # =========================================================================
    # === Interface for subclasses ============================================
    # =========================================================================

    @abstractmethod
    def _alloc_empty_ring_well_concrete(self, length, ring, well):
        '''
        Allocate an empty concrete object to store shared data for a
        ring/well pair, or the single column for the current feature.
        '''
        raise Exception("[TrialDataSharedBase][_alloc_empty_ring"\
                        "_well_concrete] Abstract method not implemented.")

    @abstractmethod
    def _alloc_empty_ring_well_last_feature_concrete(self,
                                                     length, ring, well):
        raise Exception("[TrialDataSharedBase][_alloc_empty_ring_well_last"\
                        "_feature_concrete] Abstract method not implemented.")

    # @abstractmethod
    # def _del_all_concrete(self):
    #     '''
    #     Clears all data managed by the concrete class.
    #     '''
    #     raise Exception("[TrialDataSharedBase][_del_all_concrete] "\
    #                     "Abstract method not implemented.")

    @abstractmethod
    def _get_shd(self, ring, well, chunk_slice=None):
        '''
        Returns a concrete reference to the shared data structure.
        This concrete structure have numpy index semantics.
        '''
        raise Exception("[TrialDataSharedBase][_get_shd] "\
                        "Abstract method not implemented.")

    @abstractmethod
    def _get_local(self, ring, well, chunk_slice=None):
        '''
        Returns a concrete reference to the local data structure.
        This concrete structure have numpy index semantics.
        '''
        raise Exception("[TrialDataSharedBase][_get_local] "\
                        "Abstract method not implemented.")

    @abstractmethod
    def _update_shd_col(self, ring, well, cols, data):
        '''
        Updates a set of columns on the shared data structure.
        '''
        raise Exception("[TrialDataSharedBase][_update_shd_col] "\
                        "Abstract method not implemented.")

    @abstractmethod
    def _update_local_col(self, ring, well, data):
        '''
        Updates the last column on the local data structure.
        '''
        raise Exception("[TrialDataSharedBase][_update_local_col] "\
                        "Abstract method not implemented.")

    # =========================================================================
    # === Implementations of TrialDataBase ====================================
    # =========================================================================

    def _set_ring_hook(self, ring, data):
        '''
        Add porosity and other info (coordinates and well_id) to the _data 
        storage. Adds data organized by ring and by well_id.

        Since this data is shared among all processes, it resides in shared
        memory space.
        '''

        # Allocate space for all features which should be used for training
        # for shared access.
        for w in self._wells_id_list:
            well_data = data[w]
            # Create the shared structure on all processes
            self._alloc_empty_ring_well_concrete(len(well_data), ring, w)
            
            # There may be no data for certain wells. If so, there is no
            # need to fill empty data.
            if len(well_data) == 0:
                continue

            # Copy base data to shared memory
            # The first process to acquire the lock does the work
            is_locked = self._shm_lock.acquire(blocking=False)
            if is_locked:
                field_names = [i for i, j in self._base_data_type]
                print(f'updating local r/w {ring}/{w}')
                self._update_shd_col(ring, w, field_names, well_data)

            # All processes are synced before releasing the lock. This
            # ensures that it is impossible to do the work twice since
            # the lock is only released when all processes already tried
            # to acquire it and won't try it again.
            if self._mpi_local_comm is not None:
                self._mpi_local_comm.Barrier()
            else:
                print("[TrialDataSharedBase] _mpi_local_comm is None. "\
                      "Ignore if unittesting.")

            # If the locking process reached this point, then all remaining
            # processes already forfeited the chance to copy the porosity data
            # to the shared memory.
            if is_locked:
                self._shm_lock.release()

        # Allocate space for the local single current feature
        for w in self._wells_id_list:
            well_data = data[w]
            self._alloc_empty_ring_well_last_feature_concrete(
                    len(well_data), ring, w)

    def _update_col_hook(self, r, w, feature_data):
        '''
        Updates a pair of ring/well for the last col.
        It is assumed that feature_data perfectly matches the data 
        for r and w. Thus, feature_data should have the correct size 
        of the internal data for ring r and well_id w.

        In the default case (updating the last column for a trial), 
        the data goes to local memory. If it is a commit operation, the
        data goes to shared memory.
        '''

        if self._commiting_feature:
            f_str = [f'f{self._current_feature_id}']
            self._update_shd_col(r, w, f_str, feature_data)
        else:
            self._is_last_col_empty = False
            self._update_local_col(r, w, feature_data)

    def _get_values_hook(self, r, w, chunk_slice=None):
        '''
        Return an np array with all data for a given ring and well_id.

        Data can come from both shared memory and local memory 
        (current feature), or shared memory only. On the regular case, there
        is an ongoing trial, thus data is compiled from shared and local 
        memory. When propagating, the selected features are committed, with no
        other _update_col_hook(). Thus, the last column (local) is empty and 
        should not be returned.

        The chunk_slice parameter allows the concrete class to better 
        implement its retrieval of data. If not used, all data is returned.
        '''

        # Retrieve all data
        shm_well_data = self._get_shd(r, w, chunk_slice)
        if shm_well_data.size == 0:
            # Empty ring/well case
            return np.empty(0)

        target_well_data = shm_well_data.copy()

        # Update return data with the last column on local
        # memory, if there is data on it.
        if not self._is_last_col_empty:
            target_well_data[
                    f'f{self._current_feature_id}'] = self._get_local(r,w, chunk_slice)[:]

        return target_well_data

    def _well_size_hook(self, r, w):
        '''
        Returns the number of points for a ring/well pair. Local data 
        structure is used to avoid shared structure overheads.
        '''
        return len(self._get_local(r, w))

    def _should_commit_feature_hook(self):
        '''
        Lock all processes of a node, except the first to call this method.
        Works with _done_commit_feature_hook(). After the working process
        calls _done_commit_feature_hook(), all remaining processes are 
        unlocked.
        '''

        # Shared data can only be deleted by one process.
        is_locked = self._shm_lock.acquire(blocking=False)

        # The first process to acquire the lock does the work. The remaining
        # processes wait for the working process
        if not is_locked:
            self._mpi_local_comm.Barrier()
            return False

        self._commiting_feature = True

        # Since a feature is being committed, the last column is invalid
        self._is_last_col_empty = True

        return True

    def _done_commit_feature_hook(self):
        '''
        Unlock all remaining processes and release the shm lock.
        All processes are synced before releasing the lock. This
        ensures that it is impossible to do the work twice since
        the lock is only released when all processes already tried
        to acquire it and won't try it again.
        '''

        if self._mpi_local_comm is not None:
            self._mpi_local_comm.Barrier()
        else:
            print("[TrialDataSharedBase] _mpi_local_comm is None. "\
                  "Ignore if unittesting.")

        self._commiting_feature = False
        self._shm_lock.release()
