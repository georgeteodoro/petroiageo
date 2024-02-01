import numpy as np

from TrialDataBase import TrialDataBase


class TrialDataNumpy(TrialDataBase):
    '''
    In-memory implementation of TrialDataBase using numpy as the
    concrete type for data storage.
    '''

    def __init__(self, target_wells_list, porosity_data, config):
        # Currently no initialization is needed
        super(TrialDataNumpy, self).__init__(target_wells_list, porosity_data,
                                             config)

        # Concrete data storage. Data is stored by ring. Each ring is another
        # dict by well_id
        self._data = dict()

    def _new_ring_hook(self, ring):
        '''
        A new ring is a dict of data by well_id
        '''

        self._data[ring] = dict()

    def _clear_trial_data_hook(self):
        '''
        Delete all data, resetting internal data to an empty dict.
        '''
        del self._data
        self._data = dict()

    def _set_ring_hook(self, ring, data):
        '''
        Add porosity and other info (coordinates and well_id) to the _data 
        storage. Adds data organized by ring and by well_id.
        '''

        for w in self._wells_id_list:
            well_data = data[w]
            self._data[ring][w] = np.zeros((len(well_data)),
                                           dtype=self._cur_data_type)
            field_names = [i for i, j in self._base_data_type]
            self._data[ring][w][field_names] = well_data

    def _update_col_hook(self, r, w, feature_data):
        '''
        Updates a pair of ring/well for the last col.
        It is assumed that feature_data perfectly matches the data 
        for r and w. Thus, feature_data should have the correct size 
        of the internal data for ring r and well_id w.
        '''

        f_str = f'f{self._current_feature_id}'
        self._data[r][w][f_str][:] = feature_data

    def _get_values_hook(self, r, w, chunk_slice=None):
        '''
        Should return an nparray with all data for a given ring an well_id.
        Ideally, backend data should be stored separately, since this is a
        recurrent operation.
        The chunk_slice parameter allows the concrete class to better 
        implement its retrieval of data. If not used, all data is returned.
        '''
        target_ring_data = self._data.get(r, dict())
        if len(target_ring_data) == 0:
            return np.empty(0)

        target_well_data = target_ring_data.get(w, np.empty(0))
        if not chunk_slice:
            return target_well_data
        elif target_well_data.size > 0:
            return target_well_data[chunk_slice]
        else:
            return target_well_data

    def _well_size_hook(self, r, w):
        '''
        Returns the number of points for a ring/well pair.
        '''
        return len(self._data[r].get(w, list()))
