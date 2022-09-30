import numpy as np
import h5py


def seismic_feature_np2hdf5(feature, chunk_shape):

    # Open feature
    print(f'[seismic_feature_np2hdf5] reading {feature}')
    feature_np = np.load(f'./dados/{feature}.npy')
    data_shape = feature_np.shape

    print(f'[seismic_feature_np2hdf5] writing {feature} to hdf5')
    with h5py.File(f'./dados/{feature}.h5', 'w') as h5_f:
        h5_dset = h5_f.create_dataset('f',
                                      data_shape,
                                      dtype=np.float64,
                                      chunks=chunk_shape,
                                      data=feature_np)


if __name__ == '__main__':

    features = ['NEAR', 'MID', 'FAR', 'UFAR', 'GERSZ', 'GST']
    [seismic_feature_np2hdf5(f, (100, 100, 100)) for f in features]
