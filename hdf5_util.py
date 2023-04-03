from math import ceil
import numpy as np
from lightgbm import Sequence
import numbers


# Perform a fold on a clustered h5 object, using the least amount
# of memory.
def fold_h5_all_clusters(d_h5, f, out_0):
    chunks = d_h5.chunks
    x_shape = d_h5.shape[0]
    y_shape = d_h5.shape[1]
    z_shape = d_h5.shape[2]

    out = out_0

    # Iterate on all coordinates
    for c_x in range(ceil(x_shape / chunks[0])):
        x_i = c_x * chunks[0]
        x_o = min((c_x + 1) * chunks[0], x_shape)
        for c_y in range(ceil(y_shape / chunks[1])):
            y_i = c_y * chunks[1]
            y_o = min((c_y + 1) * chunks[1], y_shape)
            for c_z in range(ceil(z_shape / chunks[2])):
                z_i = c_z * chunks[2]
                z_o = min((c_z + 1) * chunks[2], z_shape)

                # Read numpy chunk
                d_np = d_h5[x_i:x_o, y_i:y_o, z_i:z_o]

                # Append result of function and fold on out
                out = out + f(d_np)

    return out


# Perform a conditional update on a clustered h5 object, using the least amount
# of memory. All rows from the given 'column' for which the condition is True
# are updated to 'val'.
def conditional_map_h5_all_clusters(d_h5, cond_f, column_val_list):
    chunks = d_h5.chunks
    x_shape = d_h5.shape[0]
    y_shape = d_h5.shape[1]
    z_shape = d_h5.shape[2]

    # Iterate on all coordinates
    for c_x in range(ceil(x_shape / chunks[0])):
        x_i = c_x * chunks[0]
        x_o = min((c_x + 1) * chunks[0], x_shape)
        for c_y in range(ceil(y_shape / chunks[1])):
            y_i = c_y * chunks[1]
            y_o = min((c_y + 1) * chunks[1], y_shape)
            for c_z in range(ceil(z_shape / chunks[2])):
                z_i = c_z * chunks[2]
                z_o = min((c_z + 1) * chunks[2], z_shape)

                # Read numpy chunk
                d_np = d_h5[x_i:x_o, y_i:y_o, z_i:z_o]
                # print(f'=======points to update: {len(d_np[cond_f(d_np)])}')

                # Update values of each column on condition
                for column, val in column_val_list:
                    d_np[column] = np.where(cond_f(d_np), val, d_np[column])

                # Forward values to hdf5 file
                d_h5[x_i:x_o, y_i:y_o, z_i:z_o] = d_np


# Out-of-core structure to access hdf5 a dataset with chunking.
# Copying this object does not copy its data.
# This can either be a training or validation dataset (see self.train).
# Training objects shows all data, except for the ones with the given
# self.well_id.
# Validation objects only show the rows for the given self.well_id.
class HDFMultiColSequence(Sequence):

    def __init__(self, cur_h5_dset, base_features=[], train=True):
        # cur_h5_dset must be 1D
        self.cur_h5_dset = cur_h5_dset

        # should be a list of strings
        self.all_features = base_features
        self.last_col = -1

        # Indicates whether this sequence is for training
        # If for validation, only returns the subset of the
        # selected well_id instead of the remaining of points
        self.train = train

        # well_id = -1 indicates no well selected
        self.well_id = -1

        self.lenn = -1
        self.batch_size = 10000

    def update_len(self):
        self.lenn = len(self.cur_h5_dset)

    # Updates length of data as well
    def set_well_id(self, well_id):
        self.well_id = well_id
        if self.well_id == -1:
            self.lenn = len(self.cur_h5_dset)
        else:
            if self.train:
                self.lenn = len(self.cur_h5_dset[
                    self.cur_h5_dset['well_id'] != self.well_id])
            else:
                self.lenn = len(self.cur_h5_dset[self.cur_h5_dset['well_id'] ==
                                                 self.well_id])

    def set_lowo_train(self, well_id):
        self.train = True
        self.set_well_id(well_id)

    def set_lowo_val(self, well_id):
        self.train = False
        self.set_well_id(well_id)

    def index_from_coord(coord, shape):
        index = 0
        stride = 1
        for i in reversed(range(len(shape))):
            index = index + (coord[i] * stride)
            stride = stride * shape[i]

        return index

    def update_last_col_chunk(self, feature_gen, coords):
        f_str = f'f{self.last_col}'

        for f_slice, f_vals in feature_gen:
            self.cur_h5_dset[f_str, f_slice] = f_vals

    def add_new_col(self):
        self.last_col = self.last_col + 1
        f_str = f'f{self.last_col}'
        self.all_features.append(f_str)

    def get_y_np(self):
        if self.well_id == -1:
            return self.cur_h5_dset['phi']
        else:
            if self.train:
                return self.cur_h5_dset[
                    'phi', self.cur_h5_dset['well_id'] != self.well_id]
            else:
                return self.cur_h5_dset['phi', self.cur_h5_dset['well_id'] ==
                                        self.well_id]

    # Should only be used for small validation data
    def get_X_np(self):
        if self.train:
            raise Exception('[HDFMultiColSequence][get_X_np] Can only get '\
                            'data from validation sequence.')
        out_ndarray = self.cur_h5_dset[self.cur_h5_dset['well_id'] ==
                                       self.well_id][self.all_features]
        return np.array([np.array(a) for a in out_ndarray.tolist()])

    def get_item_int(self, idx):
        min_index = 0
        for cur_slice in self.cur_h5_dset.iter_chunks():
            cur_chunk = self.cur_h5_dset[cur_slice]
            if self.well_id == -1:
                well_chunk = cur_chunk
            else:
                if self.train:
                    well_chunk = cur_chunk[
                        cur_chunk['well_id'] != self.well_id]
                else:
                    well_chunk = cur_chunk[cur_chunk['well_id'] ==
                                           self.well_id]

            if (idx >= min_index) & (idx < min_index + len(well_chunk)):
                return np.array(
                    well_chunk[idx - min_index][self.all_features].tolist())
            min_index = min_index + len(well_chunk)

        raise Exception('[HDFMultiColSequence][__getitem__] Index '\
                       f'not found: {idx}.')

    def get_item_slice(self, idx):
        # print(f'[HDFMultiColSequence] getting slice {idx}')
        output = []
        min_index = 0
        for cur_slice in self.cur_h5_dset.iter_chunks():
            cur_chunk = self.cur_h5_dset[cur_slice]
            # print(cur_chunk.shape)
            if self.well_id == -1:
                well_chunk = cur_chunk
            else:
                if self.train:
                    well_chunk = cur_chunk[
                        cur_chunk['well_id'] != self.well_id]
                    # print(f'[HDFMultiColSequence] training data: '\
                    #       f'{well_chunk.shape}')
                else:
                    well_chunk = cur_chunk[cur_chunk['well_id'] ==
                                           self.well_id]
                    # print(f'[HDFMultiColSequence] validation data: '\
                    #       f'{well_chunk.shape}')

            # print(
            #     f'[HDFMultiColSequence] len(well_chunk): {len(well_chunk)}'
            # )

            # Check if the initial idx point is inside this chunk
            if (idx.start >= min_index) & (idx.start <
                                           min_index + len(well_chunk)):
                # print('[HDFMultiColSequence] initial')
                # Check if the end of the idx slice is inside this chunk
                if idx.stop <= min_index + len(well_chunk):
                    output = output + well_chunk[
                        idx.start - min_index:idx.stop -
                        min_index][self.all_features].tolist()
                    return np.array(output)

                # If not, add all points from idx.start to the end
                # of the chunk
                else:
                    output = output + well_chunk[idx.start - min_index:len(
                        well_chunk)][self.all_features].tolist()
            # Check if the initial idx point was behind, but the end point
            # is on a chunk ahead
            elif (idx.start < min_index) & (idx.stop >
                                            min_index + len(well_chunk)):
                # print('[HDFMultiColSequence] mid')
                output = output + well_chunk[:][self.all_features].tolist()
            # Otherwise, this chunk is the one with the idx stop position
            elif idx.stop <= min_index + len(well_chunk):
                # print('[HDFMultiColSequence] end')
                output = output + well_chunk[0:idx.stop - min_index][
                    self.all_features].tolist()
                return np.array(output)

            min_index = min_index + len(well_chunk)

        raise Exception('[HDFMultiColSequence][__getitem__] Couldn\'t '\
                       f'find slice: {idx}.')

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self.get_item_int(idx)
        elif isinstance(idx, slice):
            return self.get_item_slice(idx)
        else:
            raise TypeError('Sequence index must be integer, '\
                f'slice or list. Got {type(idx).__name__}')

    def __len__(self):
        return self.lenn


# Out-of-core structure to access hdf5 a dataset with chunking.
# Copying this object does not copy its data.
# Both train and validation data are inside
# Able to add new features columns on the fly as well as change a given column
class HDFMultiColList:

    # def __init__(self, cur_h5_dset, base_features=[], chunk_size=10000):
    def __init__(self, cur_h5_dset, chunk_size=10000):
        # cur_h5_dset must be 1D
        # This cur_h5_dset holds the hdf5 data
        self.cur_h5_dset = cur_h5_dset

        # should be a list of strings
        # self.all_features_and_coords = base_features
        self.all_features = []
        self.last_col = -1

        self.chunk_size = chunk_size

        self.n_chunks = cur_h5_dset.size // chunk_size + 1

    # Updates the last column with new values from a generator
    # Overwrites the previous values on this column
    def update_last_col_chunk(self, feature_gen):
        f_str = f'f{self.last_col}'

        for f_slice, f_vals in feature_gen:
            self.cur_h5_dset[f_str, f_slice] = f_vals

    # Setup a new empty last column, thus committing the current
    # last column
    def add_new_col(self):
        self.last_col = self.last_col + 1
        f_str = f'f{self.last_col}'
        # self.all_features_and_coords.append(f_str)
        self.all_features.append(f_str)

    # Returns all data, input (X) and output (y), from a given well
    def get_well_out_data(self, well_id):
        X_val_list = []
        y_val_list = []

        for cur_slice in self.cur_h5_dset.iter_chunks():
            # Load a single hypercube chunk
            cur_chunk = self.cur_h5_dset[cur_slice]

            # Filter all data which has the given well_id
            well_data = cur_chunk[cur_chunk['well_id'] == well_id]

            # Append results to output lists
            X_val_list.append(well_data[self.all_features])
            y_val_list.append(well_data['phi'])

        X_val_np = np.concatenate(X_val_list)
        y_val_np = np.concatenate(y_val_list)

        # Border case: single feature returns a (n,) shape instead of (n,1)
        if len(X_val_np.shape) == 1:
            X_val_np = np.reshape(X_val_np, (X_val_np.shape[0], 1))
            # y_val_np = np.reshape(y_val_np, (y_val_np.shape[0], 1))

        return X_val_np, y_val_np

    # Return a single chunk of data which is not related to well_id
    # No guarantees are made about the size of the output
    def get_all_well_data(self, well_id, chunk):

        # Seeks the chunk to be read
        cur_slice_it = self.cur_h5_dset.iter_chunks()
        cur_slice = cur_slice_it.__next__()
        for _ in range(chunk):
            cur_slice = cur_slice_it.__next__()

        # Load the hypercube chunk
        cur_chunk = self.cur_h5_dset[cur_slice]

        # Filter all data which is not related to the input well_id
        well_data = cur_chunk[cur_chunk['well_id'] != well_id]

        X_val_np = np.array(well_data[self.all_features])
        y_val_np = np.array(well_data['phi'])

        # Border case: single feature returns a (n,) shape instead of (n,1)
        if len(X_val_np.shape) == 1:
            X_val_np = np.reshape(X_val_np, (X_val_np.shape[0], 1))

        return X_val_np, y_val_np
