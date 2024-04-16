import numpy as np

from TrialDataSharedBase import TrialDataSharedBase


class TrialDataSharedNumpy(TrialDataSharedBase):
    '''
    '''
    def __init__(self,
                 target_wells_list,
                 porosity_data,
                 config,
                 should_consider_sampling: bool = True):
        # Currently no initialization is needed
        super(TrialDataSharedNumpy, self).__init__(
            target_wells_list, porosity_data, config, should_consider_sampling)

    def _alloc_empty_ring_well_concrete(self, length, last_feature=False):
        '''
        Allocate an empty concrete object to store shared data for a
        ring/well pair, or the single column for the current feature.
        '''

        if last_feature:
            # Only the space for a single column is allocated.
            return np.zeros((length), dtype=np.float64)
        else:
            return np.zeros((length), dtype=self._cur_data_type)

