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


def conditional_map_h5_chunk(d_h5, cond_f, column_val_list, chunk_slice):
    # Read numpy chunk
    chunk_np = d_h5[chunk_slice]

    # Update values of each column on condition
    for column, val in column_val_list:
        chunk_np[column] = np.where(cond_f(chunk_np), val, chunk_np[column])

    # Forward values to hdf5 file
    d_h5[chunk_slice] = chunk_np


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
            self.cur_h5_dset[f_str, f_slice] = np.fromiter(f_vals, np.float64)

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

        X_val_np = np.concatenate(X_val_list).reshape(-1)
        y_val_np = np.concatenate(y_val_list).reshape(-1)

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        X_val_np = np.array(X_val_np.tolist())
        y_val_np = np.array(y_val_np.tolist())

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
        X_val_np = well_data[self.all_features]
        y_val_np = well_data['phi']

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        X_val_np = np.array(X_val_np.tolist())
        y_val_np = np.array(y_val_np.tolist())

        return X_val_np, y_val_np
