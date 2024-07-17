"""
Enlarges an original numpy feature file to a larger, ANP-sized, file.

Naming convention: A region can be center (c), border (b), rod (r) or cube (q).
Example for a 2D 5x4x3 region (stack the 3 regions) 
with displacement_window = 1:

[:,:,0]       [:,:,1]       [:,:,2]
q r r r q     r b b b r     q r r r q
r b b b r     b c c c b     r b b b r
r b b b r     b c c c b     r b b b r
q r r r q     r b b b r     q r r r q
"""

import argparse
import numpy as np
from math import prod
import pathlib

import sys

sys.path.insert(0, "../..")

from alg import config_parser


def seismic_feature_np2hdf5_planar(feature_path: pathlib.Path,
                                   disp_window:int,
                                   output_dir: pathlib.Path, mult_factor):
    feature_name = feature_path.stem

    # Open feature
    print(f"[seismic_feature_np2hdf5_planar] reading {feature_name}")
    feature_np = np.load(feature_path)

    data_shape = feature_np.shape
    print(f"[seismic_feature_np2hdf5_planar] original shape: {data_shape} "
          f"with length {prod(data_shape)}")

    # Generate large data shape for replicating the data
    large_data_shape = [
        mult_factor[i] * data_shape[i] for i in range(len(data_shape))
    ]
    large_data_shape = (np.array(large_data_shape) +
                        (2 * disp_window)).tolist()

    print(f"[seismic_feature_np2hdf5_planar] new shape: {large_data_shape} "
          f"with length {prod(large_data_shape)}")

    feature_full_np = np.empty(shape=large_data_shape, dtype=np.float64)

    print(f"[seismic_feature_np2hdf5_planar] "\
          f"assigning center of {feature_name}")
    _fill_center(feature_full_np, feature_np, data_shape, mult_factor,
                 disp_window)

    print(f"[seismic_feature_np2hdf5_planar] "\
          f"assigning borders of {feature_name}")
    _fill_borders(feature_full_np, data_shape, disp_window)

    print(f"[seismic_feature_np2hdf5_planar] "\
          f"assigning cubes of {feature_name}")
    _fill_cubes(feature_full_np, feature_np, data_shape, large_data_shape,
                disp_window)

    print(f"[seismic_feature_np2hdf5_planar] "\
          f"assigning rods of {feature_name}")
    _fill_rods(feature_full_np, large_data_shape, disp_window)

    np.save(output_dir / f'{feature_name}.npy', feature_full_np, allow_pickle=False)

    

###############################################################################

def _fill_center(feature_full_np, feature_np, data_shape, mult_factor,
                 displacement_window):
    # Fill multiple copies of 'feature_np' on the large space
    for f_x in range(mult_factor[0]):
        for f_y in range(mult_factor[1]):
            for f_z in range(mult_factor[2]):
                # Create the slices shaped by the original input data
                # 'feature_np'. These slices are shifted by the displacement
                # window. For large_data these represent a single replica of
                # 'feature_np' data. x_i represents the start coordinate and
                # x_o represents the end coordinate.
                x_i = displacement_window + (f_x * data_shape[0])
                x_o = displacement_window + ((f_x + 1) * data_shape[0])
                y_i = displacement_window + (f_y * data_shape[1])
                y_o = displacement_window + ((f_y + 1) * data_shape[1])
                z_i = displacement_window + (f_z * data_shape[2])
                z_o = displacement_window + ((f_z + 1) * data_shape[2])

                # Fill values of the current slice with a
                # full copy of feature_np
                feature_full_np[x_i:x_o, y_i:y_o, z_i:z_o] = feature_np[:, :, :]


def _fill_borders(feature_full_np, data_shape, displacement_window):
    """
    Create the slices for the region inside the displacement window. This
    region have the feature_np data (with mult replications if needed).
    However, it excludes the displacement windows borders.
    """
    x_slice = slice(displacement_window, data_shape[0] - displacement_window)
    y_slice = slice(displacement_window, data_shape[1] - displacement_window)
    z_slice = slice(displacement_window, data_shape[2] - displacement_window)

    # Top/bottom regions
    for z in range(displacement_window):
        # Top
        feature_full_np[x_slice, y_slice, z] = feature_full_np[x_slice, y_slice,
                                                               0]

        # Bottom
        feature_full_np[x_slice, y_slice, data_shape[2] - z -
                        1] = feature_full_np[x_slice, y_slice,
                                             data_shape[2] - 1]

    # Left/right regions
    for y in range(displacement_window):
        # Left
        feature_full_np[x_slice, y, z_slice] = feature_full_np[x_slice, 0,
                                                               z_slice]

        # Right
        feature_full_np[x_slice, data_shape[1] - y - 1,
                        z_slice] = feature_full_np[x_slice, data_shape[1] - 1,
                                                   z_slice]

    # Front/back regions
    for x in range(displacement_window):
        # Front
        feature_full_np[x, y_slice, z_slice] = feature_full_np[0, y_slice,
                                                               z_slice]

        # Back
        feature_full_np[data_shape[0] - x - 1, y_slice,
                        z_slice] = feature_full_np[data_shape[0] - 1, y_slice,
                                                   z_slice]


def _fill_cubes(feature_full_np, feature_np, data_shape, large_data_shape,
                displacement_window):
    # Create slices for displacement border regions
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


def _fill_rods(feature_full_np, large_data_shape, displacement_window):
    """
    Create the slices for the region inside the displacement window. This
    region have the feature_np data (with mult replications if needed).
    However, it excludes the displacement windows borders.
    """
    x_slice = slice(displacement_window,
                    large_data_shape[0] - displacement_window)
    y_slice = slice(displacement_window,
                    large_data_shape[1] - displacement_window)
    z_slice = slice(displacement_window,
                    large_data_shape[2] - displacement_window)

    # Top rod regions
    for y in range(displacement_window):
        for z in range(displacement_window):
            # Top left
            feature_full_np[
                displacement_window:large_data_shape[0] - displacement_window,
                y,
                z,
            ] = feature_full_np[x_slice, 0, 0]

            # Top right
            feature_full_np[
                displacement_window:large_data_shape[0] - displacement_window,
                large_data_shape[1] - displacement_window + y,
                z,
            ] = feature_full_np[x_slice, large_data_shape[1] - 1, 0]

            x = y
            # Top front
            feature_full_np[
                x,
                displacement_window:large_data_shape[1] - displacement_window,
                z,
            ] = feature_full_np[0, y_slice, 0]

            # Top back
            feature_full_np[
                large_data_shape[0] - displacement_window + x,
                displacement_window:large_data_shape[1] - displacement_window,
                z,
            ] = feature_full_np[large_data_shape[0] - 1, y_slice, 0]

    # Center rod regions
    for x in range(displacement_window):
        for y in range(displacement_window):
            # Front left
            feature_full_np[
                x,
                y,
                displacement_window:large_data_shape[2] - displacement_window,
            ] = feature_full_np[0, 0, z_slice]

            # Front right
            feature_full_np[
                x,
                large_data_shape[1] - displacement_window + y,
                displacement_window:large_data_shape[2] - displacement_window,
            ] = feature_full_np[0, large_data_shape[1] - 1, z_slice]

            # Back left
            feature_full_np[
                large_data_shape[0] - displacement_window + x,
                y,
                displacement_window:large_data_shape[2] - displacement_window,
            ] = feature_full_np[large_data_shape[0] - 1, 0, z_slice]

            # Back right
            feature_full_np[
                large_data_shape[0] - displacement_window + x,
                large_data_shape[1] - displacement_window + y,
                displacement_window:large_data_shape[2] - displacement_window,
            ] = feature_full_np[large_data_shape[0] - 1,
                                large_data_shape[1] - 1, z_slice]

    # Bottom rod regions
    for y in range(displacement_window):
        for z in range(displacement_window):
            # Bottom left
            feature_full_np[
                displacement_window:large_data_shape[0] - displacement_window,
                y,
                large_data_shape[2] - displacement_window + z,
            ] = feature_full_np[x_slice, 0, large_data_shape[2] - 1]

            # Bottom right
            feature_full_np[
                displacement_window:large_data_shape[0] - displacement_window,
                large_data_shape[1] - displacement_window + y,
                large_data_shape[2] - displacement_window + z,
            ] = feature_full_np[x_slice, large_data_shape[1] - 1,
                                large_data_shape[2] - 1]

            x = y
            # Bottom front
            feature_full_np[
                x,
                displacement_window:large_data_shape[1] - displacement_window,
                large_data_shape[2] - displacement_window + z,
            ] = feature_full_np[0, y_slice, large_data_shape[2] - 1]

            # Bottom back
            feature_full_np[
                large_data_shape[0] - displacement_window + x,
                displacement_window:large_data_shape[1] - displacement_window,
                large_data_shape[2] - displacement_window + z,
            ] = feature_full_np[large_data_shape[0] - 1, y_slice,
                                large_data_shape[2] - 1]


###############################################################################


def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument(
        '-f',
        dest='feature_dir',
        type=pathlib.Path,
        required=True,
        help="The base features directory to read files from.",
    )

    parser.add_argument(
        '-o',
        dest='output_dir',
        type=pathlib.Path,
        required=False,
        help="Output directory for newly created h5 files. If not defined, "\
        "-f feature_dir is used.",
    )

    parser.add_argument(
        '--large',
        dest='mult_factor',
        action='store',
        required=True,
        help="Return a larger dataset for testing. the 'mult_factor' "
        "represents how much larger the original hypercube should be."
        "It should be a tuple of 3 values, each multiplying one of the"
        "dimensions (x, y, z). E.g., (1, 2, 2).",
    )

    parser.add_argument(
        '--config',
        dest='config_file_path',
        action='store',
        required=True,
        help=
        "The YAML config file path with wells coords. This is necessary to garantee"
        " the wells coords ordering.",
    )

    return parser


if __name__ == '__main__':
    parser = config_arg_parser()
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.feature_dir

    # Base mult_factor is 1, thus returning the original data
    # without enlarging it
    mult_factor = (1, 1, 1)
    if args.mult_factor:
        mult_factor = eval(args.mult_factor)

    print(args.output_dir, type(args.output_dir))

    base_features_dir = pathlib.Path(args.feature_dir)

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
        # 'FAR.npy',
        'all'
    ]
    if target_features_files_names_with_extension[0] == "all":
        complete_files_path = list(base_features_dir.glob("*.npy"))
    else:
        complete_files_path = [
            base_features_dir / file
            for file in target_features_files_names_with_extension
        ]

    for file in complete_files_path:
        if not file.exists():
            raise FileExistsError(f"File {file} doesn't exists!")

        if not file.is_file():
            raise ValueError(f"{file} is not a file!")

        if not file.suffix == '.npy':
            raise ValueError(f"{file} is not a .npy file!")

    disp_window = config_parser.YAMLConfig(
        args.config_file_path).wells['window']
    print(f"[LOG] Disp window found: {disp_window}")

    [
        seismic_feature_np2hdf5_planar(f, disp_window, args.output_dir,
                                       mult_factor) for f in complete_files_path
    ]
