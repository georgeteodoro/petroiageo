"""
This script transforms .npy features data to hdf5 format so it can be used by 
the algorithm.
"""

import argparse
import numpy as np
import h5py
from math import prod
import pathlib


def coord_3d_to_planar(x, y, z, shape):
    return x * shape[2] * shape[1] + y * shape[2] + z


def seismic_feature_np2hdf5_planar(feature_path: pathlib.Path, chunk_shape,
                                   displacement_window,
                                   target_folder: pathlib.Path):

    feature_name = feature_path.stem

    # Open feature
    print(f'[seismic_feature_np2hdf5_planar] reading {feature_name}')
    feature_np = np.load(feature_path)

    data_shape = feature_np.shape

    print(f'[seismic_feature_np2hdf5_planar] original shape: {data_shape} '\
          f'with length {prod(data_shape)}')

    large_data_shape = create_new_3d_np_array_with_borders(
        displacement_window, data_shape)

    print(f'[seismic_feature_np2hdf5_planar] new shape: {large_data_shape} '\
          f'with length {prod(large_data_shape)}')

    feature_full_np = np.empty(shape=large_data_shape, dtype=np.float64)

    x_slice, y_slice, z_slice = create_slices_for_internal_region_of_feature_full_np(
        displacement_window, large_data_shape)

    feature_full_np = assign_regular_inside_points(
        feature_name, feature_np, feature_full_np, x_slice, y_slice, z_slice)

    print(
        f'[seismic_feature_np2hdf5_planar] assigning borders of {feature_name}')
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


# <<<<<<< HEAD:src/data/features/seismic_data3_hdf5.py
    create_feature_hdf5_file(feature_name, target_folder, chunk_shape,
                             large_data_shape, feature_full_np)


def assign_regular_inside_points(feature_name, feature_np, feature_full_np,
                                 x_slice, y_slice, z_slice):
    print(
        f'[seismic_feature_np2hdf5_planar] assigning center of {feature_name}')
    feature_full_np[x_slice, y_slice, z_slice] = feature_np[:, :, :]
    return feature_full_np


def create_slices_for_internal_region_of_feature_full_np(
    displacement_window, large_data_shape):
    x_slice = slice(displacement_window,
                    large_data_shape[0] - displacement_window)
    y_slice = slice(displacement_window,
                    large_data_shape[1] - displacement_window)
    z_slice = slice(displacement_window,
                    large_data_shape[2] - displacement_window)

    return x_slice, y_slice, z_slice


def create_feature_hdf5_file(feature_name: str, target_folder: pathlib.Path,
                             chunk_shape, large_data_shape, feature_full_np):
    print(
        f'[seismic_feature_np2hdf5_planar] creating hdf5 of feature {feature_name}'
    )
    with h5py.File(target_folder / f'{feature_name}.h5', 'w') as h5_f:
        _ = h5_f.create_dataset('f',
                                large_data_shape,
                                dtype=np.float64,
                                chunks=chunk_shape,
                                data=feature_full_np.flat)


def create_new_3d_np_array_with_borders(displacement_window, data_shape):
    return (np.array(data_shape) + (2 * displacement_window)).tolist()


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


def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument('--f-dir',
                        dest='feat_folder',
                        type=pathlib.Path,
                        required=True,
                        help="Input dir with .npy base features.")

    parser.add_argument(
        '-o',
        dest='target_folder',
        type=pathlib.Path,
        required=False,
        help="Output dir. Default is the same as the input dir.")
    return parser


if __name__ == '__main__':

    parser = config_arg_parser()
    args = parser.parse_args()

    if args.target_folder is None:
        args.target_folder = args.feat_folder

    print(args.target_folder, type(args.target_folder))

    base_features_folder = pathlib.Path(args.feat_folder)

    target_features_files_names_with_extension = [
        # 'NEAR.npy', 'NEAR_envelope_.npy', 'NEAR_gersztenkorn_5-5-9.npy',
        # 'NEAR_instantaneous-frequency_.npy', 'NEAR_rms-5_.npy', 'MID.npy',
        # 'NEAR_gaussian-curvature_.npy', 'NEAR_gst_3-3-11.npy',
        # 'NEAR_max-curvature_.npy', 'NEAR_shape-index_.npy',
        # 'NEAR_azimuth_.npy', 'NEAR_gersztenkorn_3-3-11.npy',
        # 'NEAR_gst_3-3-7.npy', 'NEAR_mean-curvature_.npy',
        # 'NEAR_sobel_5-5-11.npy', 'NEAR_contour-curvature_.npy',
        # 'NEAR_gersztenkorn_3-3-7.npy', 'NEAR_gst_3-3-9.npy',
        # 'NEAR_min-curvature_.npy', 'UFAR.npy', 'NEAR_curvedness_.npy',
        # 'NEAR_gersztenkorn_3-3-9.npy', 'NEAR_gst_5-5-11.npy',
        # 'NEAR_most-negative-curvature_.npy', 'NEAR_dip-angle_.npy',
        # 'NEAR_gersztenkorn_5-5-11.npy', 'NEAR_gst_5-5-7.npy',
        # 'NEAR_most-positive-curvature_.npy', 'NEAR_dip-curvature_.npy',
        # 'NEAR_gersztenkorn_5-5-7.npy', 'NEAR_gst_5-5-9.npy',
        'FAR.npy'
    ]

    complete_files_path = [
        base_features_folder / file
        for file in target_features_files_names_with_extension
    ]

    for file in complete_files_path:
        if not file.exists():
            raise FileExistsError(f"File {file} doesn't exists!")

        if not file.is_file():
            raise ValueError(f"{file} is not a file!")

        if not file.suffix == ".npy":
            raise ValueError(f"{file} is not a .npy file!")

    disp_window = 3
    chunk_shape = (100, 100, 251 + disp_window + disp_window)
    [
        seismic_feature_np2hdf5_planar(f, chunk_shape, disp_window,
                                       args.target_folder)
        for f in complete_files_path
    ]
