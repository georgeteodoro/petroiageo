import numpy as np
import h5py
import os

from TrialDataBase import TrialDataBase

TMP_DSET_NAME = 'c'


class TrialDataH5(TrialDataBase):
    '''
    Out-of-core implementation of TrialDataBase using HDF5 as the
    concrete type for data storage. The OOC backend means that only filtered
    data shall reside in memory. Thus, it is possible to simply load all data
    in memory or for the H5 file to reside in virtual memory's file cache.
    '''

    def __init__(self,
                 target_wells_list,
                 porosity_data,
                 config,
                 should_consider_sampling: bool = True):
        super(TrialDataH5, self).__init__(target_wells_list, porosity_data,
                                          config, should_consider_sampling)

        # Define H5 filename
        mpi_local_comm = config.get_param('mpi_local_comm')
        rank = 0
        if mpi_local_comm is not None:
            rank = mpi_local_comm.Get_rank()
        self._filename = f"/tmp/TD-tmp-r{rank}.h5"

        # If the cur file exists, it should be deleted
        if os.path.exists(self._filename):
            os.remove(self._filename)

        # Create H5 file
        self._cur_h5 = h5py.File(f'{self._filename}', 'w')

    def _new_ring_hook(self, ring):
        '''
        Creation of datasets is done by ring,well pair, not just ring.
        Does nothing.
        '''

        pass

    def _clear_trial_data_hook(self):
        '''
        Delete all data by deleting the H5 file
        '''

        if os.path.exists(self._filename):
            os.remove(self._filename)

        # Create new H5 file
        self._cur_h5 = h5py.File(f'{self._filename}', 'w')

    def _set_ring_hook(self, ring, data, overwrite=False):
        '''
        Add porosity and other info (coordinates and well_id) to the H5 
        storage file. Adds data organized by ring and by well_id.
        If there is no dataset yet, creates one.
        '''

        for w in self._wells_id_list:
            well_data = data[w]

            # Check if a new dataset should be created
            dset_name = f'r{ring}-w{w}'
            dset = self._cur_h5.get(dset_name)
            if overwrite and dset is not None:
                del dset
                dset = None
            if dset is None:
                dset = self._cur_h5.create_dataset(dset_name,
                                                   (len(well_data), ),
                                                   dtype=self._cur_data_type)

            # Copy data to H5 storage
            field_names = [i for i, j in self._base_data_type]
            for i, field in enumerate(field_names):
                dset[field] = [item[i] for item in well_data]

    def _update_col_hook(self, r, w, feature_data):
        '''
        Updates a pair of ring/well for the last col.
        It is assumed that feature_data perfectly matches the data 
        for r and w. Thus, feature_data should have the correct size 
        of the internal data for ring r and well_id w.
        '''

        # Get dset for the ring,well pair
        dset_name = f'r{r}-w{w}'
        dset = self._cur_h5.get(dset_name)
        assert dset is not None

        f_str = f'f{self._current_feature_id}'
        dset[f_str] = feature_data

    def _get_values_hook(self, r, w, chunk_slice=None):
        '''
        Should return an nparray with all data for a given ring an well_id.
        Ideally, backend data should be stored separately, since this is a
        recurrent operation. I.e., should not return a reference to the 
        actual data, instead returning a copy.
        The chunk_slice parameter allows the concrete class to better 
        implement its retrieval of data. If not used, all data is returned.
        '''
        # target_ring_data = self._data.get(r, dict())
        # if len(target_ring_data) == 0:
        #     return np.empty(0)

        # Get dset for the ring,well pair
        dset_name = f'r{r}-w{w}'
        dset = self._cur_h5.get(dset_name)

        if dset is None:
            return np.empty(0)
        else:
            if chunk_slice is None:
                return dset[:].copy()
            else:
                return dset[chunk_slice].copy()

        # target_well_data = target_ring_data.get(w, np.empty(0))
        # if not chunk_slice:
        #     return target_well_data.copy()
        # elif target_well_data.size > 0:
        #     return target_well_data[chunk_slice].copy()
        # else:
        #     return target_well_data.copy()

    def _well_size_hook(self, r, w):
        '''
        Returns the number of points for a ring/well pair.
        '''

        dset_name = f'r{r}-w{w}'
        dset = self._cur_h5.get(dset_name)
        assert dset is not None

        return len(dset)
