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

    def _set_ring_hook(self, ring, data):
        '''
        Add porosity and other info (coordinates and well_ifd) to a trial_data 
        ring, updating internally its size
        '''

        self._trial_data_dict[ring] = np.zeros((len(data)),
                                               dtype=self._cur_data_type)
        fields = [i for i, j in self._base_data_type]
        self._trial_data_dict[ring][fields] = data

    def _update_col_from_ring_hook(self, r, feature_data):
        f_str = f'f{self._current_feature_id}'
        self._trial_data_dict[r][f_str][:] = feature_data

    def _get_ring_filtered_values_hook(self, ring_key, chunk_id, well_filter):
        # Each chunk should have a ratio of ring points proportional to
        # the number of chunks.
        n_training_chunks = int(
            self._config.alg['parallel']['n_training_chunks'])
        all_ring_points = self._trial_data_dict[ring_key]
        training_chunk_size = len(all_ring_points) / n_training_chunks

        # Get all points from current ring, filtered by a well_filter
        chunk_slice = slice(int(chunk_id * training_chunk_size),
                            int((chunk_id + 1) * training_chunk_size))
        new_points = well_filter(all_ring_points[chunk_slice])
        return new_points

    def _in_well_filter_hook(self, well_id):
        return lambda data: data[data['well_id'] == well_id]

    def _not_in_well_filter_hook(self, well_id):
        if (well_id is None) or (well_id < 0):
            # If well_id=-1 then no filtering is required
            well_filter = lambda data: data
        else:
            # Otherwise, returns all points outside well_id
            well_filter = lambda data: data[data['well_id'] != well_id]

        return well_filter

    def _get_ring_np_values_hook(self, ring):
        return self._trial_data_dict[ring]