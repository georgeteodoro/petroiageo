"""
This script generates the main input seismic/porosity cube to be used by the 
algorithm. It requires the wells coordinates information along with the 
wells porosity file.
"""
import argparse
import h5py
from math import prod
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm

import sys

sys.path.insert(0, "..")

from alg import common, config_parser


def _new_random_coord(x_len, y_len, expanded_real_points):
    f_rand = np.random.uniform

    # Workaround for lack of do-while
    new_x = expanded_real_points[0][0]
    new_y = expanded_real_points[0][1]

    # Get a new coordinate for the expanded point
    while (new_x, new_y) in expanded_real_points:
        new_x = abs(f_rand()) * x_len
        new_y = abs(f_rand()) * y_len

    return round(new_x), round(new_y)


def porosity_points_py2hdf5(porosity_file: str, hdf5_file_path: str,
                            config_file_path: pathlib.Path,
                            feat_file_path: str, mult_factor: tuple):

    print(f"[porosity_points_py2hdf5] Loading config file")
    real_points, disp_window = get_wells_coords_and_disp_window(
        config_file_path)
    print(f"[porosity_points_py2hdf5] Wells coords found:\n{real_points}")
    print(f"[porosity_points_py2hdf5] Disp window found:\n{disp_window}")

    # Hypercube shape from feature file contains the padding for the
    # displacement. This is unnecessary here
    feat_file_path = pathlib.Path(feat_file_path)
    if feat_file_path.suffix == ".h5":
        # hypercube_shape = h5py.File(pathlib.Path(feat_file_path),
        # 'r')['f'].shape
        raise ValueError("A feature of extension .h5 is no longer supported!" +
                         " In this version, it must be a npy file!")
    elif feat_file_path.suffix == ".npy":
        hypercube_shape = np.load(feat_file_path).shape
    else:
        raise ValueError("Feature file extension must be .npy " +
                         f"but got {feat_file_path.suffix} instead!")

    # Enlarge shape if required
    if mult_factor is not None:
        original_hypercube_shape = hypercube_shape
        print(f"[porosity_points_py2hdf5] Initial shape: {hypercube_shape}")
        hypercube_shape = [a * b for a, b in zip(hypercube_shape, mult_factor)]

    print(f"[porosity_points_py2hdf5] Hypercube shape: {hypercube_shape}")

    chunk_shape = (100, 100, hypercube_shape[2])
    print(f"[porosity_points_py2hdf5] Chunk shape: {chunk_shape}")

    print("[porosity_points_py2hdf5] Creating hdf5 file")
    new_h5_porosity_path = pathlib.Path(hdf5_file_path)
    new_h5_porosity_path.parent.mkdir(exist_ok=True, parents=True)
    porosity_h5_f = h5py.File(new_h5_porosity_path, 'w')
    data_type = np.dtype([
        ('x', np.int64),
        ('y', np.int64),
        ('z', np.int64),
        ('phi', np.float64),
        ('real', np.int64),
        ('ring', np.int64),
        ('well_id', np.int64),
    ])
    porosity_h5_dset = porosity_h5_f.create_dataset(
        common.POROSITY_DSET_NAME,
        hypercube_shape,
        dtype=data_type,
        chunks=chunk_shape,
    )
    print(
        f"[porosity_points_py2hdf5] Target cube shape: {porosity_h5_dset.shape}"
    )

    # If true, it will not commit data, just check if the propper chunks are 
    # being generated
    simulate = False

    if not simulate:
        print(f"[porosity_points_py2hdf5] Filling coordinates")
        (x_len, y_len, z_len) = hypercube_shape
        for i in tqdm(range(x_len)):
            # Batching of yz coordinates for writing on hdf5 file
            all_yz = []
            for j in range(y_len):
                # Batching of z coordinates for writing on hdf5 file
                all_z = []
                for k in range(z_len):
                    all_z = all_z + [(i, j, k, 0, common.RealValues.empty, -1, -1)]
                all_yz = all_yz + [all_z]
            # Commit all points for a given x coordinate
            porosity_h5_dset[i, ...] = all_yz

        print(f"[porosity_points_py2hdf5] Loading porosity file")
        porosity_file_path = pathlib.Path(porosity_file)
        porosity_np = np.load(porosity_file_path)
        porosity_np = np.core.records.fromarrays(porosity_np.transpose(),
                                                 names='x, y, z, phi',
                                                 formats='i8, i8, i8, f8')

    if mult_factor is not None:
        print(f"[porosity_points_py2hdf5] Adding extra "\
               "well points for large data")

        # Create a list of chunks of the initial size
        # large_chunks = []
        large_chunks_xy = []
        for x in range(mult_factor[0]):
            for y in range(mult_factor[1]):
                large_chunks_xy.append((x, y))
                # for z in range(mult_factor[2]):
                #     large_chunks.append((x, y, z))
        print(f'chunks to generate: {large_chunks_xy}')

    if not simulate:
        # Copy all porosity initial points to h5
        print(f"[porosity_points_py2hdf5] Updating {len(porosity_np)} values")
        for row in tqdm(porosity_np.tolist()):
            x, y, z, p = row
            ring = 0

            if (x, y) in real_points:
                real = common.RealValues.real
                well_id = real_points.index((x, y))
            else:
                real = common.RealValues.canal
                well_id = -1

            porosity_h5_dset[x, y, z] = (x, y, z, p, real, ring, well_id)

    # Copy-paste the whole initial sub-region multiple times, if mult_factor
    if mult_factor is not None:
        new_wells_coords = real_points.copy()
        # large_chunks.remove((0, 0, 0))
        # print(f"[porosity_points_py2hdf5] multi {len(large_chunks)} chunks")
        # print(large_chunks)
        (x_len, y_len, z_len) = original_hypercube_shape
        # for chk_z in tqdm(range(mult_factor[2]),
        #                   total=mult_factor[2],
        #                   position=1):
            # print(f'z level: {chk_z}')

        for count, (chk_x, chk_y) in tqdm(enumerate(large_chunks_xy),
                                          total=len(large_chunks_xy),
                                          position=0):
            # if (chk_x, chk_y, chk_z) == (0, 0, 0):
            if (chk_x, chk_y) == (0, 0):
                # This is the original chunk, no work required
                continue

            # print(f"updating {chk_x, chk_y, chk_z}")
            xi = chk_x * x_len
            yi = chk_y * y_len

            xo = (chk_x + 1) * x_len
            yo = (chk_y + 1) * y_len

            # print(f'[{xi}:{xo}, {yi}:{yo}, {zi}:{zo}]')
            # print(f'[0:{x_len}, 0:{y_len}, 0:{z_len}]')
            # print('')
            # return

            print(f'large {chk_x},{chk_y}, well_id: {count * len(real_points)}')

            if not simulate:
                # Copy base info, one depth segment at a time
                for chk_z in range(mult_factor[2]):
                    zi = chk_z * z_len
                    zo = (chk_z + 1) * z_len

                    porosity_h5_dset[xi:xo, yi:yo, zi:zo, 'phi', 'real', 'ring',
                                     'well_id'] = porosity_h5_dset[0:x_len,
                                                                   0:y_len,
                                                                   0:z_len, 'phi',
                                                                   'real', 'ring',
                                                                   'well_id']
                # Remaining updates are for the full depth
                zi = 0
                zo = hypercube_shape[2]

                # Increment well_id
                porosity_h5_dset[xi:xo, yi:yo, zi:zo,
                                 'well_id'] += count * len(real_points)

                # Reset well_id from empty points
                wid_tmp_real = porosity_h5_dset[xi:xo, yi:yo, zi:zo, 'real']
                wid_tmp_well_id = porosity_h5_dset[xi:xo, yi:yo, zi:zo,
                                                   'well_id']

                wid_tmp_well_id[wid_tmp_real != common.RealValues.real] = -1
                porosity_h5_dset[xi:xo, yi:yo, zi:zo,
                                 'well_id'] = wid_tmp_well_id

            # Do nothing about coordinates since all points are filled first

            # Add new well coords
            new_wells_coords += [
                tuple([x + y for x, y in zip(a, b)])
                for a, b in zip(real_points, [(xi, yi)] * len(real_points))
            ]

        print("New wells for large hypercube:")
        print(new_wells_coords)

    porosity_h5_f.close()


def get_wells_coords_and_disp_window(
        config_file_path: pathlib.Path) -> tuple[list, int]:
    config = config_parser.YAMLConfig(config_file_path)
    real_points = config.wells_as_simple_list
    window = config.wells['window']
    return real_points, window


def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument(
        '-f',
        dest='feat_file_path',
        action='store',
        required=True,
        help="A base h5 feat file path to get the hypercube shape from. "
        "If used with --large, this feature should be with the original "
        "size, not the large one or one with the displacement padding.",
    )
    parser.add_argument(
        '-p',
        dest='porosity_file',
        action='store',
        required=True,
        help="Numpy list of real points. Each line represents (x, y, z, phi).")

    parser.add_argument(
        '--config',
        dest='config_file_path',
        action='store',
        required=True,
        help="The YAML config file path with wells coords. This is necessary "
        "to guarantee the wells coords ordering.",
    )

    parser.add_argument(
        '-o',
        dest='hdf5_file',
        action='store',
        required=True,
        help="The hdf5 file path to be generated.",
    )

    parser.add_argument(
        '--large',
        dest='mult_factor',
        action='store',
        required=False,
        help="Return a larger dataset for testing. the 'mult_factor' "
        "represents how much larger the original hypercube should be."
        "It should be a tuple of 3 values, each multiplying one of the"
        "dimensions (x, y, z). E.g., (1, 2, 2).",
    )

    return parser


if __name__ == '__main__':

    parser = config_arg_parser()
    args = parser.parse_args()

    porosity_file_path = args.porosity_file
    hdf5_file_path = args.hdf5_file
    if args.mult_factor:
        mult_factor = eval(args.mult_factor)
    config_file_path = pathlib.Path(args.config_file_path)
    feat_file_path = args.feat_file_path

    porosity_points_py2hdf5(
        porosity_file_path,
        hdf5_file_path,
        config_file_path,
        feat_file_path,
        mult_factor,
    )
