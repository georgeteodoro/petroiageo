from TestDataBase import TestDataBase 

class TestDataNumpy(TestDataBase):
	'''
	'''

	def __init__(self, arg):
		super(TestDataNumpy, self).__init__()
		self.arg = arg

    def _create_new_ring_hook(self):
        return np.empty((_ring_size(it, depth)), dtype=self._cur_data_type)

    def _append_ring_hook(self, ring, data):
        '''
        Add data to a test_data ring, updating internally its size
        '''
        filt_list = self._f_sel_filter.satisfies(data)
        filt_data = data[filt_list]

        size = len(filt_data)
        beg = self._test_data_size[ring]
        end = beg + size
        self._test_data_size[ring] += size

        if self._features_only:
            self._test_data_dict[ring][
                'x',
                'y',
                'z',
                'phi',
                beg:end,
            ] = filt_data[['x', 'y', 'z', 'phi']]
        else:
            self._test_data_dict[ring][
                'x',
                'y',
                'z',
                'phi',
                'well_id',
                beg:end,
            ] = filt_data[['x', 'y', 'z', 'phi', 'well_id']]

    def _update_col_from_ring_hook(self, r, feature_data):
        f_str = f'f{self._current_feature_id}'
        self._test_data_dict[r][f_str][:] = feature_data

    def _get_ring_filtered_values_hook(self, r, well_filter):
        # Get all points from current ring, filtered by well_id
        ring_points = self._test_data_dict[r]
        new_points = well_filter(ring_points)
        return new_points

    def _in_well_filter_hook(self, well_id):
        return lambda p: data[data['well_id'] == well_id]


    def _not_in_well_filter_hook(self, well_id):
        if not well_id or well_id < 0:
            # If well_id=-1 then no filtering is required
            well_filter = lambda data: data
        else:
            # Otherwise, returns all points outside well_id
            well_filter = lambda data: data[data['well_id'] != well_id]

        return well_filter