from abc import ABC, abstractmethod
from h5py import Dataset
import numpy as np
from numpy.lib import recfunctions as rfn
from time import time
from math import ceil, prod
from tqdm import tqdm

import common
from config_parser import Config
from data_filter import WellsSingleRingDataFilter
from sampler import ChunkSamplerV1


class TrialDataNumpyNotHier(ABC):
    '''
    Concrete non-hierarchical TrialData.
    All data is stored in the same numpy array and filtered.

    All trial_data should be an array of the following columns, on this order:
     - x, y, z (point coordinates)
     - phi (porosity)
     - well_id (only required for feature selection, not propagation)
     - features (multiple columns)
    Due to the use of padding, all coordinates are the padded coordinates. 
    Thus, it is expected of the wells_list to have padded coordinates as well.
    '''

    def __init__(self,
                 target_wells_ids_list,
                 porosity_data: Dataset,
                 config: Config):

        self._config = config

        self._n_features = config.alg['max_num_features']

        # Set the datatype for points
        self._base_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('well_id', np.int64),
            ('real', np.int64),
        ]
        self._cur_data_type = self._base_data_type + [
            (f'f{f}', np.float64) for f in range(self._n_features + 1)
        ]
        self._cur_data_type = np.dtype(self._cur_data_type)

        # Internal state
        self._current_feature_id = -1
        self._current_ring = -1
        self._current_features = []

        # # List of all rings' IDs which are present on the concrete backend
        # # data structures. This allows concrete subclasses not to worry about
        # # this common data, while this superclass can still know which rings
        # # were successfully added to the concrete object.
        # self._rings_list = []

        # This is the list target wells. These can be training or test wells.
        self._wells_id_list = target_wells_ids_list
        self._f_sel_filter = WellsSingleRingDataFilter(target_wells_ids_list)

        self._porosity_data = porosity_data

        # All trial data
        self._data = None

    # =========================================================================
    # === Public interface ====================================================
    # =========================================================================

    def prepare_porosity(self, prep_it: int):
        '''
        Allocate data required for the current iteration it.
        Fill coordinates, phi and well_id (when necessary).
        It also resets the internal current column.
        First iteration is 0.

        prep_it: Integer representing the iteration whose data we should 
                 prepare. For running iteration 20 (will propagate ring 20)
                 prep_it should be 19.
        '''

        profile = self._config.get_param('prof_trial_prep_porosity')

        t0 = time()

        assert prep_it>=0, f"[TrialDataNumpyNotHier][prepare_porosity] "\
            f"First iteration is 0, but received current iteration {prep_it}."


        # Set the first feature, even if it's empty
        self._current_feature_id = 0
        self._current_features = []
        self._current_features.append(f'f{self._current_feature_id}')

        # Get total number of points
        # self._porosity_data[self._porosity_data['ring'] <= prep_it]
        n_points = 0
        n_chunks = zip(self._porosity_data.shape, self._porosity_data.chunks)
        n_chunks = [a/b for a,b in n_chunks]
        n_chunks = [ceil(a) for a in n_chunks]
        n_chunks = prod(n_chunks)

        # for chunk_slice in tqdm(self._porosity_data.iter_chunks(), 
        #         total=n_chunks):
        for chunk_slice in self._porosity_data.iter_chunks():
            chunk_np = self._porosity_data[chunk_slice]
            # 0/0
            cur_points = (chunk_np['ring'] <= prep_it) & (
                chunk_np['ring'] >= 0)
            cur_points = sum(sum(sum(cur_points)))
            n_points += cur_points

        # print(n_points)

        # Allocate array
        self._data = np.zeros((n_points), dtype=self._cur_data_type)

        # Copy data
        cur_init = 0
        field_names = [i for i, j in self._base_data_type]
        # for chunk_slice in tqdm(self._porosity_data.iter_chunks(), 
        #         total=n_chunks):
        for chunk_slice in self._porosity_data.iter_chunks():
            # Load porosity data
            chunk_np = self._porosity_data[chunk_slice]

            # Filter porosity data from porosity chunk
            filt_data = chunk_np[(chunk_np['ring'] <= prep_it) &
                (chunk_np['ring'] >= 0)]

            # print(filt_data)
            # print(filt_data.shape)

            # print(self._data)
            # print(self._data.shape)

            # Update values on local data
            self._data[cur_init:cur_init+len(filt_data)][
                field_names] = filt_data[
                ['x', 'y', 'z', 'phi', 'well_id', 'real',]]

            cur_init += len(filt_data)

        

    def commit_feature(self, feature, disp):
        '''
        Commits the current feature, then setting up the next feature.
        '''
        assert self._current_feature_id >= 0, "[TrialDataNumpyNotHier] "\
            "[commit_feature] Committing feature before prepare_porosity."
        assert self._current_feature_id < self._n_features, \
            "[TrialDataNumpyNotHier][commit_feature] Committing beyond " \
            f"last feature: cur_feature={self._current_feature_id} "\
            f"n_features={self._n_features}."

        self.update_feature(feature, disp)

        self._current_feature_id += 1
        if self._current_feature_id < self._n_features:
            self._current_features.append(f'f{self._current_feature_id}')

    def update_feature(self, feature, disp):
        profile = self._config.get_param('prof_trial_update_feature')

        
        f_str = f'f{self._current_feature_id}'
        # print('getting coords')
        coords = self._data[['x','y','z']]
        # print('filtering feature_data')
        feature_data = feature.filter_coords(coords)
        # print('updating feature')

        self._data[f_str] = feature_data

    def get_train_values(self, well_id, chunk_id):
        '''
        Leave-one-well-out validation function. Returns all data that
        is NOT on well_id. If well_id=-1, then all data is returned.
        Generates the training data inplace. X_train and y_train are
        generated inplace to avoid reallocation for them.
        If no out-of-core is used internally, then chunk_id=0 and all
        trial_data is returned (filtered by well_id obviously).
        '''

        # wells_to_retrieve is a list of indices
        wells_to_retrieve = list(self._wells_id_list)
        if well_id >= 0:
            wells_to_retrieve.remove(well_id)
        return self._get_values(wells_to_retrieve, chunk_id)

    def get_val_values(self, well_id):
        wells_to_retrieve = [well_id]
        # chunk_id=0 to return all data
        return self._get_values(wells_to_retrieve, -1)

    def get_num_wells(self):
        return len(self._wells_id_list)

    def __len__(self):
        return len(self._data)

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

    def _get_values(self, wells_to_retrieve, chunk_id):
        '''
        Helper function for filtering trial_data.
        '''

        # Chunking is not implemented...
        assert chunk_id < 1

        
        well_cond = np.full(self._data.shape, False)
        for w in wells_to_retrieve:
            well_cond |= self._data['well_id'] == w

        filtered_values = self._data[well_cond]

        # print(wells_to_retrieve)
        # print(sum(well_cond))

        # Remove fields and change 1D fielded array to 2D array
        X = rfn.structured_to_unstructured(
            filtered_values[self._current_features])
        y = filtered_values['phi']

        return X, y
