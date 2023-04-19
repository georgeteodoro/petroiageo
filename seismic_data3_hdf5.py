import numpy as np
import h5py
from math import prod


def coord_3d_to_planar(x, y, z, shape):
    return x * shape[2] * shape[1] + y * shape[2] + z


def seismic_feature_np2hdf5_planar(feature, chunk_shape, displacement_window):

    # Open feature
    print(f'[seismic_feature_np2hdf5_planar] reading {feature}')
    feature_np = np.load(f'./dados/{feature}.npy')

    # displacement_window = 2
    # feature_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # feature_np[0, 0, 0] = 1
    # feature_np[1, 0, 0] = 2
    # feature_np[0, 1, 0] = 3
    # feature_np[1, 1, 0] = 4
    # feature_np[0, 0, 1] = 5
    # feature_np[0, 1, 1] = 6
    # feature_np[1, 0, 1] = 7
    # feature_np[1, 1, 1] = 8
    data_shape = feature_np.shape

    print(f'[seismic_feature_np2hdf5_planar] original shape: {data_shape} '\
          f'with length {prod(data_shape)}')

    # Create new 3d np array with borders
    large_data_shape = (np.array(data_shape) +
                        (2 * displacement_window)).tolist()

    print(f'[seismic_feature_np2hdf5_planar] new shape: {large_data_shape} '\
          f'with length {prod(large_data_shape)}')

    feature_full_np = np.empty(shape=large_data_shape, dtype=np.float64)
    # feature_full_np = np.empty(shape=large_data_shape, dtype=int)

    # Create slices for the internal region of feature_full_np
    x_slice = slice(displacement_window,
                    large_data_shape[0] - displacement_window)
    y_slice = slice(displacement_window,
                    large_data_shape[1] - displacement_window)
    z_slice = slice(displacement_window,
                    large_data_shape[2] - displacement_window)

    # Assign regular inside points
    print(f'[seismic_feature_np2hdf5_planar] assigning center of {feature}')
    feature_full_np[x_slice, y_slice, z_slice] = feature_np[:, :, :]

    print(f'[seismic_feature_np2hdf5_planar] assigning borders of {feature}')
    # Top/bottom regions
    for z in range(displacement_window):
        feature_full_np[x_slice, y_slice, z] = feature_np[:, :, 0]
        feature_full_np[x_slice, y_slice, large_data_shape[2] - z -
                        1] = feature_np[:, :, data_shape[2] - 1]

    # Left/right regions
    for y in range(displacement_window):
        feature_full_np[x_slice, y, z_slice] = feature_np[:, 0, :]
        feature_full_np[x_slice, large_data_shape[1] - y - 1,
                        z_slice] = feature_np[:, data_shape[1] - 1, :]

    # Front/back regions
    for x in range(displacement_window):
        feature_full_np[x, y_slice, z_slice] = feature_np[0, :, :]
        feature_full_np[large_data_shape[0] - x - 1, y_slice,
                        z_slice] = feature_np[data_shape[0] - 1, :, :]

    # Create slices for regions
    top_slice = slice(0, displacement_window)
    bottom_slice = slice(large_data_shape[2] - displacement_window,
                         large_data_shape[2])
    left_slice = slice(0, displacement_window)
    right_slice = slice(large_data_shape[1] - displacement_window,
                        large_data_shape[1])
    front_slice = slice(0, displacement_window)
    back_slice = slice(large_data_shape[0] - displacement_window,
                       large_data_shape[0])

    # Top-front-left cube region
    feature_full_np[front_slice, left_slice, top_slice] = feature_np[0, 0, 0]

    # Top-front-right cube region
    feature_full_np[front_slice, right_slice,
                    top_slice] = feature_np[0, data_shape[1] - 1, 0]

    # Top-back-left cube region
    feature_full_np[back_slice, left_slice,
                    top_slice] = feature_np[data_shape[0] - 1, 0, 0]

    # Top-back-right cube region
    feature_full_np[back_slice, right_slice,
                    top_slice] = feature_np[data_shape[0] - 1,
                                            data_shape[1] - 1, 0]

    # Bottom-front-left cube region
    feature_full_np[front_slice, left_slice,
                    bottom_slice] = feature_np[0, 0, data_shape[2] - 1]

    # Bottom-front-right cube region
    feature_full_np[front_slice, right_slice,
                    bottom_slice] = feature_np[0, data_shape[1] - 1,
                                               data_shape[2] - 1]

    # Bottom-back-left cube region
    feature_full_np[back_slice, left_slice,
                    bottom_slice] = feature_np[data_shape[0] - 1, 0,
                                               data_shape[2] - 1]

    # Bottom-back-right cube region
    feature_full_np[back_slice, right_slice,
                    bottom_slice] = feature_np[data_shape[0] - 1,
                                               data_shape[1] - 1,
                                               data_shape[2] - 1]

    # Top rod regions
    for y in range(displacement_window):
        for z in range(displacement_window):
            # Top left
            feature_full_np[displacement_window:large_data_shape[0] -
                            displacement_window, y, z] = feature_np[:, 0, 0]

            # Top right
            feature_full_np[displacement_window:large_data_shape[0] -
                            displacement_window,
                            large_data_shape[1] - displacement_window + y,
                            z] = feature_np[:, data_shape[1] - 1, 0]

            x = y
            # Top front
            feature_full_np[x, displacement_window:large_data_shape[1] -
                            displacement_window, z] = feature_np[0, :, 0]

            # Top back
            feature_full_np[large_data_shape[0] - displacement_window + x,
                            displacement_window:large_data_shape[1] -
                            displacement_window,
                            z] = feature_np[data_shape[0] - 1, :, 0]

    # Center rod regions
    for x in range(displacement_window):
        for y in range(displacement_window):
            # Front left
            feature_full_np[x, y, displacement_window:large_data_shape[2] -
                            displacement_window] = feature_np[0, 0, :]

            # Front right
            feature_full_np[x, large_data_shape[1] - displacement_window + y,
                            displacement_window:large_data_shape[2] -
                            displacement_window] = feature_np[0,
                                                              data_shape[1] -
                                                              1, :]

            # Back left
            feature_full_np[large_data_shape[0] - displacement_window + x, y,
                            displacement_window:large_data_shape[2] -
                            displacement_window] = feature_np[data_shape[0] -
                                                              1, 0, :]

            # Back right
            feature_full_np[large_data_shape[0] - displacement_window + x,
                            large_data_shape[1] - displacement_window + y,
                            displacement_window:large_data_shape[2] -
                            displacement_window] = feature_np[data_shape[0] -
                                                              1,
                                                              data_shape[1] -
                                                              1, :]

    # Bottom rod regions
    for y in range(displacement_window):
        for z in range(displacement_window):
            # Bottom left
            feature_full_np[displacement_window:large_data_shape[0] -
                            displacement_window, y,
                            large_data_shape[2] - displacement_window +
                            z] = feature_np[:, 0, data_shape[2] - 1]

            # Bottom right
            feature_full_np[displacement_window:large_data_shape[0] -
                            displacement_window,
                            large_data_shape[1] - displacement_window + y,
                            large_data_shape[2] - displacement_window +
                            z] = feature_np[:, data_shape[1] - 1,
                                            data_shape[2] - 1]

            x = y
            # Bottom front
            feature_full_np[x, displacement_window:large_data_shape[1] -
                            displacement_window,
                            large_data_shape[2] - displacement_window +
                            z] = feature_np[0, :, data_shape[2] - 1]

            # Bottom back
            feature_full_np[large_data_shape[0] - displacement_window + x,
                            displacement_window:large_data_shape[1] -
                            displacement_window,
                            large_data_shape[2] - displacement_window +
                            z] = feature_np[data_shape[0] - 1, :,
                                            data_shape[2] - 1]

    # Create hdf5 file
    print(
        f'[seismic_feature_np2hdf5_planar] creating hdf5 of feature {feature}')
    with h5py.File(f'./dados/{feature}.h5', 'w') as h5_f:
        # h5_dset = h5_f.create_dataset('f', (prod(large_data_shape), ),
        #                               dtype=np.float64,
        #                               chunks=(prod(chunk_shape), ),
        #                               data=feature_full_np.flat)
        h5_dset = h5_f.create_dataset('f', large_data_shape,
                                      dtype=np.float64,
                                      chunks=chunk_shape,
                                      data=feature_full_np.flat)


# NOD DONE YET (may be unused)
def seismic_feature_np2hdf5_3d(feature, chunk_shape, max_displacement):

    # Open feature
    print(f'[seismic_feature_np2hdf5] reading {feature}')
    feature_np = np.load(f'./dados/{feature}.npy')
    data_shape = feature_np.shape

    # Add max_displacement to begin and end of every coordinate
    data_shape[0] = data_shape[0] + 2 * max_displacement
    data_shape[1] = data_shape[1] + 2 * max_displacement
    data_shape[2] = data_shape[2] + 2 * max_displacement

    # Create hdf5 file
    print(f'[seismic_feature_np2hdf5] writing {feature} to hdf5')
    with h5py.File(f'./dados/{feature}.h5', 'w') as h5_f:
        h5_dset = h5_f.create_dataset(
            'f',
            (prod(data_shape), ),
            dtype=np.float64,
            chunks=(prod(chunk_shape), ),
            # data=feature_np.flat
        )

    # Fill the data in the center of the structure


if __name__ == '__main__':

    # features = ['NEAR', 'MID', 'FAR', 'UFAR', 'GERSZ', 'GST']
    features = ['FAR']
    # features = [
    #     "FAR", "MID", "NEAR_azimuth_", "NEAR_contour-curvature_",
    #     "NEAR_curvedness_", "NEAR_dip-angle_", "NEAR_dip-curvature_",
    #     "NEAR_envelope_", "NEAR_gaussian-curvature_",
    #     "NEAR_gersztenkorn_3-3-11", "NEAR_gersztenkorn_3-3-7",
    #     "NEAR_gersztenkorn_3-3-9", "NEAR_gersztenkorn_5-5-11",
    #     "NEAR_gersztenkorn_5-5-7", "NEAR_gersztenkorn_5-5-9",
    #     "NEAR_gst_3-3-11", "NEAR_gst_3-3-7", "NEAR_gst_3-3-9",
    #     "NEAR_gst_5-5-11", "NEAR_gst_5-5-7", "NEAR_gst_5-5-9",
    #     "NEAR_instantaneous-frequency_", "NEAR_max-curvature_",
    #     "NEAR_mean-curvature_", "NEAR_min-curvature_",
    #     "NEAR_most-negative-curvature_", "NEAR_most-positive-curvature_",
    #     "NEAR", "NEAR_rms-5_", "NEAR_shape-index_", "NEAR_sobel_5-5-11", "UFAR"
    # ]

    disp_window = 3
    [
        seismic_feature_np2hdf5_planar(
            f, (100, 100, 251 + disp_window + disp_window), disp_window)
        for f in features
    ]
