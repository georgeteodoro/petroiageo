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

data_shape = (10, 20, 30)
chunk_shape = (10, 10, 10)
n_real_points = 1000

# data_shape = (1000, 200, 300)
# chunk_shape = (100, 100, 100)
# n_real_points = 20000

data_len = prod(data_shape)
n_real_wells = 5


def coord_from_index(idx, shape):
    c = []
    rolling_idx = idx
    for i in range(len(shape)):
        stride = prod(shape[i + 1:])
        coord = int(floor(rolling_idx / stride))
        rolling_idx = rolling_idx - coord * stride
        c.append(coord)

    return tuple(c)


def create_h5_feature(name):
    with h5py.File(f'f_{name}.h5', 'w') as h5_f:
        # Prepare numpy synthetic data
        data_np = np.empty(data_shape, dtype=np.float64)
        print('Creating random feature points')
        for i in tqdm(range(data_shape[0])):
            for j in range(data_shape[1]):
                for k in range(data_shape[2]):
                    data_np[i, j, k] = random()

        h5_dset = h5_f.create_dataset('f',
                                      data_shape,
                                      dtype=np.float64,
                                      chunks=chunk_shape,
                                      data=data_np)


def create_h5_porosity():
    with h5py.File(f'porosity_full.h5', 'w') as h5_f:
        data_type = np.dtype([('x', np.int64), ('y', np.int64),
                              ('z', np.int64), ('phi', np.float64),
                              ('real', np.int64), ('well_id', np.int64)])

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


# From https://stackoverflow.com/questions/15182381
def fields_view(arr, fields):
    dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


# From LGB example dataset_from_multi_hdf5.py
class HDFMultiColTrainSequence(lgb.Sequence):

    def __init__(self, porosity_h5, features_dict_h5, batch_size, train,
                 well_out):
        """
        Construct a sequence object from HDF5 with required interface.
        Parameters
        ----------
        hdf_dataset : h5py.Dataset
            Dataset in HDF5 file.
        batch_size : int
            Size of a batch. When reading data to construct lightgbm 
            Dataset, each read reads batch_size rows.
        """
        # We can also open HDF5 file once and get access to
        # self.data = hdf_dataset
        self.batch_size = batch_size

        self.porosity_h5 = porosity_h5
        self.features_dict_h5 = features_dict_h5
        self.selected_features = []
        self.cur_feature = ""

        self.train = train
        self.well_out = well_out

    # Updates the current feature to be used besides the previously
    # selected features
    def set_current_feature(self, cur_feature):
        # Check if this feature wasn't already added
        if cur_feature in self.selected_features:
            raise Exception('[HDFMultiColSequence] Feature '\
                f'{cur_feature} already added.')
        if cur_feature == self.cur_feature:
            raise Exception('[HDFMultiColSequence] Feature '\
                f'{cur_feature} was already the current feature.')

        self.cur_feature = cur_feature

    # Pushes the current feature into the selected features.
    # Fails if there ware no selected features
    def push_current_feature(self):
        if len(cur_feature) == 0:
            raise Exception('[HDFMultiColSequence] Pushing empty feature.')

        self.selected_features.append(self.cur_feature)
        self.cur_feature = ''

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            nd_idx = coord_from_index(idx, self.porosity_h5.shape)
            data = self.porosity_h5[nd_idx]
            coordinates = [data['x'], data['y'], data['z']]
            features = [
                self.features_dict_h5[f][nd_idx]
                for f in self.selected_features + [self.cur_feature]
            ]
            # print(coordinates + features)
            return np.array(coordinates + features)
        elif isinstance(idx, slice):
            # Loading a simple slice and then linearly slicing it as a np array
            # Can be optimized later to read as little as possible on
            # the big_slice
            min_x = int(
                floor(idx.start /
                      (porosity_h5.shape[1] * porosity_h5.shape[2])))
            max_x = int(1 +
                        floor(idx.stop /
                              (porosity_h5.shape[1] * porosity_h5.shape[2])))

            # Create output array
            # All points are float to enable the flattening of data on a
            # single np array
            all_features = self.selected_features + [self.cur_feature]
            output = np.empty((idx.stop - idx.start, 3 + len(all_features)),
                              dtype=np.float64)

            # Find coordinates for the flat big_slice
            first_idx_3d = coord_from_index(idx.start, porosity_h5.shape)
            first_idx = porosity_h5.shape[2] * first_idx_3d[1] + first_idx_3d[2]

            # Get coordinates
            big_slice = self.porosity_h5[min_x:max_x, :, :]
            accurate_slice = big_slice.flat[first_idx:first_idx + idx.stop -
                                            idx.start]

            # print(f'doing slices: {idx}')
            # print(f'shape: {porosity_h5.shape}')
            # print(f'big_slice shape: {big_slice.shape}')
            # print(f'acc_slice shape: {accurate_slice.shape}')
            # print(f'first idx3d: {first_idx_3d}')
            # print(f'first idx: {first_idx}')

            output[:, 0] = accurate_slice['x']
            output[:, 1] = accurate_slice['y']
            output[:, 2] = accurate_slice['z']

            # Update the values for each feature column
            i = 3
            for f in all_features:
                big_slice = self.features_dict_h5[f][min_x:max_x, :, :]
                output[:, i] = big_slice.flat[first_idx:first_idx + idx.stop -
                                              idx.start]
                i = i + 1

            return output
        elif isinstance(idx, list):
            return self.porosity_h5[[coord_from_index(i) for i in idx]]
        else:
            raise TypeError('Sequence index must be integer, '\
                f'slice or list. Got {type(idx).__name__}')

    def __len__(self):
        # if self.train:
        #     all_data = self.porosity_h5[
        #         self.porosity_h5['well_id'] != self.well_out]
        # else:
        #     all_data = self.porosity_h5[self.porosity_h5['well_id'] ==
        #                                 self.well_out]
        return prod(self.porosity_h5.shape)


if __name__ == '__main__':
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

    # Prepare hdf5 sequences objects
    train_seq = HDFMultiColTrainSequence(porosity_h5, {
        'feature1': f1_h5,
        'feature2': f2_h5
    },
                                         prod(chunk_shape),
                                         train=True,
                                         well_out=well_out)
    train_seq.set_current_feature('feature1')

    val_seq = HDFMultiColTrainSequence(porosity_h5, {
        'feature1': f1_h5,
        'feature2': f2_h5
    },
                                       prod(chunk_shape),
                                       train=False,
                                       well_out=well_out)

    # lgb doesn't accept label sequence
    # label_seq = HDFLabelsSequence(porosity_h5, prod(chunk_shape))
    # lgb_train_dataset = lgb.Dataset(train_seq, label=label_seq)

    # Prepare label points with 1 well out
    label_np = porosity_h5['phi']
    # label_np = porosity_h5[porosity_h5['well_id'] != well_out]['phi']
    # label_val_np = porosity_h5[porosity_h5['well_id'] == well_out]['phi']

    # Prepare LGB datasets
    lgb_train_dataset = lgb.Dataset(train_seq, label_np.ravel())
    # lgb_eval_dataset = lgb.Dataset(eval_seq,
    #                                label_val_np.ravel,
    #                                reference=lgb_train_dataset)

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

    # Perform training
    print('training...')
    regressor = lgb.train(
        params,
        lgb_train_dataset,
        num_boost_round=100,
        # valid_sets=lgb_eval_dataset,
        # callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

    # Create multi-sequence X data:
