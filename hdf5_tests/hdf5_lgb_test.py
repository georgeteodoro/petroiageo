import h5py
import numpy as np
from math import prod, floor, ceil
import gc
import psutil, os, sys
from random import random
import importlib
import numbers
import lightgbm as lgb
from tqdm import tqdm
from time import time

# data_shape = (10, 20, 30)
# chunk_shape = (10, 10, 10)
# n_real_points = 1000

data_shape = (1000, 200, 300)
chunk_shape = (100, 100, 100)
n_real_points = 20000

data_len = prod(data_shape)
n_real_wells = 5

data_type = np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64),
                      ('phi', np.float64), ('real', np.int64),
                      ('well_id', np.int64)])

max_features = 2


def coord_from_index(idx, shape):
    c = []
    rolling_idx = idx
    for i in range(len(shape)):
        stride = prod(shape[i + 1:])
        coord = int(floor(rolling_idx / stride))
        rolling_idx = rolling_idx - coord * stride
        c.append(coord)

    return tuple(c)


def index_from_coord(coord, shape):
    index = 0
    stride = 1
    for i in reversed(range(len(shape))):
        index = index + (coord[i] * stride)
        stride = stride * shape[i]

    return index


def create_h5_feature(name):
    with h5py.File(f'f_{name}.h5', 'w') as h5_f:
        # Prepare numpy synthetic data
        data_np = np.empty(data_shape, dtype=np.float64)
        print('Creating random feature points')
        for i in tqdm(range(data_shape[0])):
            for j in range(data_shape[1]):
                for k in range(data_shape[2]):
                    data_np[i, j, k] = random()

        h5_dset = h5_f.create_dataset('f', (prod(data_shape), ),
                                      dtype=np.float64,
                                      chunks=(prod(chunk_shape), ),
                                      data=data_np)


def create_h5_porosity():
    with h5py.File(f'porosity_full.h5', 'w') as h5_f:
        # Prepare numpy synthetic data
        data_np = np.empty(data_shape, dtype=data_type)

        # Fill all points with coordinates and random porosity
        print('Creating random porosity points')
        for i in tqdm(range(data_shape[0])):
            for j in range(data_shape[1]):
                for k in range(data_shape[2]):
                    data_np[int(i), int(j), int(k)] = (i, j, k, random(), 0, 0)

        # Set some of them as 'real' points
        print('Setting real points on porosity dataset')
        for n in tqdm(range(n_real_points)):
            i = int(floor(random() * data_shape[0]))
            j = int(floor(random() * data_shape[1]))
            k = int(floor(random() * data_shape[2]))

            # Re-roll values for already real points
            while data_np[i, j, k]['real'] == 1:
                i = int(floor(random() * data_shape[0]))
                j = int(floor(random() * data_shape[1]))
                k = int(floor(random() * data_shape[2]))

            data_np[i, j, k] = (i, j, k, random(), 1,
                                int(floor(random() * n_real_wells)))

        h5_dset = h5_f.create_dataset('p',
                                      data_shape,
                                      dtype=data_type,
                                      chunks=chunk_shape,
                                      data=data_np)


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


# From LGB example dataset_from_multi_hdf5.py
class HDFMultiColSequence(lgb.Sequence):

    def __init__(self, cur_dset_h5, base_features, train):
        # cur_dset_h5 must be 1D
        self.cur_dset_h5 = cur_dset_h5

        # should be a list of strings
        self.all_features = base_features

        # Indicates whether this sequence is for training
        # If for validation, only returns the subset of the
        # selected well_id instead of the remaining of points
        self.train = train
        self.well_id = -1
        self.lenn = -1
        self.batch_size = 10000

    # Updates length of data as well
    def set_well_id(self, well_id):
        self.well_id = well_id
        if self.train:
            self.lenn = len(
                self.cur_dset_h5[self.cur_dset_h5['well_id'] != self.well_id])
        else:
            self.lenn = len(
                self.cur_dset_h5[self.cur_dset_h5['well_id'] == self.well_id])

    def add_feature(self, f_str):
        self.all_features.append(f_str)

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            min_index = 0
            for cur_slice in self.cur_dset_h5.iter_chunks():
                cur_chunk = self.cur_dset_h5[cur_slice]
                if self.train:
                    well_chunk = cur_chunk[
                        cur_chunk['well_id'] != self.well_id]
                else:
                    well_chunk = cur_chunk[cur_chunk['well_id'] ==
                                           self.well_id]

                if (idx >= min_index) & (idx < min_index + len(well_chunk)):
                    return np.array(
                        well_chunk[idx -
                                   min_index][self.all_features].tolist())
                min_index = min_index + len(well_chunk)
        elif isinstance(idx, slice):
            output = []
            min_index = 0
            for cur_slice in self.cur_dset_h5.iter_chunks():
                cur_chunk = self.cur_dset_h5[cur_slice]
                if self.train:
                    well_chunk = cur_chunk[
                        cur_chunk['well_id'] != self.well_id]
                else:
                    well_chunk = cur_chunk[cur_chunk['well_id'] ==
                                           self.well_id]

                # Check if the initial idx point is inside this chunk
                if (idx.start >= min_index) & (idx.start <
                                               min_index + len(well_chunk)):
                    # Check if the end of the idx slice is inside this chunk
                    if idx.stop < min_index + len(well_chunk):
                        output = output + well_chunk[
                            idx.start - min_index:idx.stop -
                            min_index][self.all_features].tolist()
                        break
                    # If not, add all points from idx.start to the end
                    # of the chunk
                    else:
                        output = output + well_chunk[idx.start - min_index:len(
                            well_chunk)][self.all_features].tolist()
                # Check if the initial idx point was behind, but the end point
                # is on a chunk ahead
                elif (idx.start < min_index) & (idx.stop >
                                                min_index + len(well_chunk)):
                    output = output + well_chunk[:][self.all_features].tolist()
                # Otherwise, this chunk is the one with the idx stop position
                else:
                    output = output + well_chunk[0:idx.stop - min_index][
                        self.all_features].tolist()
                    return np.array(output)

                min_index = min_index + len(well_chunk)

        else:
            raise TypeError('Sequence index must be integer, '\
                f'slice or list. Got {type(idx).__name__}')

    def __len__(self):
        return self.lenn


def add_feature_to_dset(cur_h5_dset, features):
    # Fill current dataset features values
    for (f_str, f_h5) in features:
        for chunk_slice in cur_h5_dset.iter_chunks():
            # Get coordinates of the current dataset slice
            coords = cur_h5_dset['x', 'y', 'z', chunk_slice[0]]

            # Get list of points to be updated
            linear_coords = [
                index_from_coord(c, data_shape) for c in coords.flat
            ]
            linear_coords.sort()

            # Update feature values
            cur_h5_dset[f_str, chunk_slice[0]] = f_h5[linear_coords]


def read_only_expanded():
    t0 = time()
    # Create synthetic data
    create_h5_feature('feature1')
    create_h5_feature('feature2')
    create_h5_porosity()

    well_out = 1

    # Open data files
    porosity_h5 = h5py.File('porosity_full.h5', 'r')['p']
    real_points = fold_h5_all_clusters(porosity_h5,
                                       lambda d: len(d[d['real'] == 1]), 0)
    f1_h5 = h5py.File('f_feature1.h5', 'r')['f']
    f2_h5 = h5py.File('f_feature2.h5', 'r')['f']

    print(f'real points: {real_points}')

    cur_data_type = [('x', np.int64), ('y', np.int64), ('z', np.int64),
                     ('phi', np.float64), ('well_id', np.int64)]
    cur_data_type = cur_data_type + [(f'f{f}', np.float64)
                                     for f in range(max_features)]
    cur_data_type = np.dtype(cur_data_type)
    print(cur_data_type)

    cur_h5 = h5py.File('cur.h5', 'w')
    cur_chunksize = prod(data_shape) / prod(chunk_shape) * 10
    cur_h5_dset = cur_h5.create_dataset('c', (real_points, ),
                                        dtype=cur_data_type,
                                        chunks=(cur_chunksize, ))

    # Fill current dataset porosity values
    prev_end = 0
    for x_c, y_c, z_c in porosity_h5.iter_chunks():
        # Get current chunk
        chunk_np = porosity_h5[x_c, y_c, z_c]

        # Filter only one type of points
        real_points = chunk_np[chunk_np['real'] == 1]

        # Assign these porosity values to the current dataset
        cur_h5_dset['x', 'y', 'z', 'phi', 'well_id',
                    prev_end:(prev_end + len(real_points))] = real_points[[
                        'x', 'y', 'z', 'phi', 'well_id'
                    ]]
        prev_end = prev_end + len(real_points)

    t1 = time()

    # Set training Sequence objects
    x_train_seq = HDFMultiColSequence(cur_h5_dset, ['x', 'y', 'z'], True)
    x_val_seq = HDFMultiColSequence(cur_h5_dset, ['x', 'y', 'z'], False)
    

    # add_feature_to_dset(cur_h5_dset, [('f0', f1_h5), ('f1', f2_h5)])
    add_feature_to_dset(cur_h5_dset, 0, [('f0', f1_h5)])
    x_train_seq.add_feature('f0')
    x_val_seq.add_feature('f0')

    print(len(cur_h5_dset))
    print(cur_h5_dset[20:24])

    t2 = time()

    params = {
        "max_bin": 128,
        "max_depth": 10,
        "learning_rate": 0.1,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 20,
        "verbose": -1,
        "min_data": 10,
        "boost_from_average": True,
        "bagging_freq": 1,
        "random_state": 0,
        # "tree_learner": "data",
    }

    for well_out in [0,1,2]:
        t21 = time()
        # Update well out on sequence objects
        x_train_seq.set_well_id(well_out)
        x_val_seq.set_well_id(well_out)

        # Update y values for the current well out
        y_train_np = cur_h5_dset[cur_h5_dset['well_id'] != well_out]['phi']
        y_val_np = cur_h5_dset[cur_h5_dset['well_id'] == well_out]['phi']

        print(f'y_train_np: {y_train_np.shape}')

        # Generate datasets for the current well out
        lgb_train_dataset = lgb.Dataset(x_train_seq, y_train_np)
        lgb_eval_dataset = lgb.Dataset(x_val_seq, y_val_np)
        
        # Perform training
        print('training...')
        regressor = lgb.train(
            params,
            lgb_train_dataset,
            num_boost_round=100,
            valid_sets=lgb_eval_dataset,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        t22 = time()
        print(f'Trained well_out {well_out} in {t22-t21}')

    t3 = time()

    print(f'Data-gen-time: {t1-t0}')
    print(f'Dataset-setup-time: {t2-t1}')
    print(f'Training-time: {t3-t2}')

if __name__ == '__main__':
    read_only_expanded()
