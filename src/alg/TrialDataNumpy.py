import numpy as np

from TrialDataBase import TrialDataBase


class TrialDataNumpy(TrialDataBase):
    '''
    In-memory implementation of TrialDataBase using numpy as the
    concrete type for data storage.
    '''
    def __init__(self, features_only, f_sel_filter, porosity_data, config):
        # Currently no initialization is needed
        super(TrialDataNumpy, self).__init__(features_only, f_sel_filter,
                                             porosity_data, config)

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

        for w in range(len(self._wells_list)):
            well_data = data[w]
            self._data[ring][w] = np.zeros((len(well_data)),
                                           dtype=self._cur_data_type)
            fields = [i for i, j in self._base_data_type]
            self._data[ring][w][fields] = well_data

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
        
        if not chunk_slice:
            return self._data[r][w]
        else:
            return self._data[r][w][chunk_slice]

    def _well_size_hook(self, r, w):
        '''
        Returns the number of points for a ring/well pair.
        '''
        return len(self._data[r][w])


    # def _get_ring_filtered_values_hook(self, ring_key, chunk_id, well_filter):
    #     # Each chunk should have a ratio of ring points proportional to
    #     # the number of chunks. Fractions are rounded up, so the last
    #     # chunk can have fewer elements.
    #     n_training_chunks = int(
    #         self._config.alg['parallel']['n_training_chunks'])
    #     training_chunk_size = int(
    #         ceil(len(all_ring_points) / n_training_chunks))

    #     # We cannot know ahead how many non-test/validation points are withing
    #     # a slice. Thus, we keep getting new slices until the expected number
    #     # of points is reached.
    #     new_points = []

    #     # Begin by filtering the current ring
    #     all_ring_points = self._trial_data_dict[ring_key]
    #     new_points = well_filter(all_ring_points)

    #     # Get all points from current ring, filtered by a well_filter
    #     chunk_slice = slice(int(chunk_id * training_chunk_size),
    #                         int((chunk_id + 1) * training_chunk_size))
    #     return new_points

    # def _get_ring_np_values_hook(self, ring):
    #     return self._trial_data_dict[ring]