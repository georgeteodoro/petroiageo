"""
This script generates the main input seismic/porosity cube to be used by the 
algorithm. It requires the wells coordinates information along with the 
wells porosity file.
"""
import argparse
import h5py
from math import prod
import numpy as np
import pathlib
from tqdm import tqdm

import sys

sys.path.insert(0, "..")

from alg import common


def _new_random_coord(x, y, expanded_real_points):
    f_rand = np.random.normal

    # Workaround for lack of do-while
    new_x = expanded_real_points[0][0]
    new_y = expanded_real_points[0][1]

    # Get a new coordinate for the expanded point
    while (new_x, new_y) in expanded_real_points:
        new_x = f_rand() * x
        new_y = f_rand() * y

    return new_x, new_y


def porosity_points_py2hdf5(porosity_file, hdf5_file_path, hypercube_shape,
                            chunk_shape, real_points, mult_factor):
    print(f"[porosity_points_py2hdf5] Loading porosity file")
    porosity_file_path = pathlib.Path(porosity_file)
    # if porosity_file_path.suffix == ".txt":
    #     porosity_np = np.loadtxt(porosity_file_path, delimiter=" ")
    # elif porosity_file_path.suffix == ".npy":
    porosity_np = np.load(porosity_file_path, allow_pickle=True)

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
        'p',
        hypercube_shape,
        dtype=data_type,
        chunks=chunk_shape,
    )

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

    print(f"[porosity_points_py2hdf5] Updating {len(porosity_np)} values")
    for x, y, z, p in tqdm(porosity_np):
        if (x, y) in real_points:
            real = common.RealValues.real
            well_id = real_points.index((x, y))
            ring = 0
        else:
            real = common.RealValues.canal
            well_id = -1
            ring = -1
        porosity_h5_dset[np.int64(x), np.int64(y),
                         np.int64(z)] = (
                             np.int64(x),
                             np.int64(y),
                             np.int64(z),
                             p,
                             real,
                             ring,
                             well_id,
                         )

    if mult_factor > 1:
        print(f"[porosity_points_py2hdf5] Adding extra "\
               "well points for large data")

        # Create a sample full depth well
        full_depth_well = np.empty(z_len, dtype=data_type)
        count = 0
        for x, y, z, p in porosity_np:
            if (x, y) in real_points:
                full_depth_well[count] = (
                    np.int64(x),
                    np.int64(y),
                    np.int64(count),
                    p,
                    common.RealValues.real,
                    0,
                    -1,
                )
                count += 1
                if count == z_len:
                    break

        # Filter only real data
        expanded_real_points = real_points.copy()
        well_id = len(real_points)
        for _ in range(1, mult_factor * len(real_points)):
            # Get a new random coordinate for the well and update it in the
            # full_depth_well to be copied
            new_x, new_y = _new_random_coord(x, y, expanded_real_points)
            full_depth_well['x'] = new_x
            full_depth_well['y'] = new_y
            full_depth_well['well_id'] = well_id
            well_id += 1

            # Add new well to h5 dataset
            porosity_h5_dset[np.int64(x), np.int64(y), :] = full_depth_well[:]

    porosity_h5_f.close()


def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument(
        '-f',
        dest='feat_file_path',
        action='store',
        required=True,
        help="A base h5 feat file path to get the hypercube shape from",
    )
    parser.add_argument(
        '-p',
        dest='porosity_file',
        action='store',
        required=True,
        help="The file path with the base porosity. As it may be a subset "
        "of the whole 3D cube, it is not possible to get the cube shape "
        "from this.",
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
        default=1,
        help="Augment the number of well ponts by MULT_FACTOR. E.g., for "
        "MULT_FACTOR=3, all real well points are replicated 2 times on random"
        "locations in addition to the original location.",
    )

    return parser


if __name__ == '__main__':
    wells_coords = [
	(34, 97), (98, 46), (55, 30), (101, 132), (33, 184)
    ]

    parser = config_arg_parser()
    args = parser.parse_args()
    hypercube_shape = h5py.File(pathlib.Path(args.feat_file_path),
                                'r')['f'].shape
    porosity_file_path = args.porosity_file
    hdf5_file_path = args.hdf5_file
    mult_factor = int(args.mult_factor)

    chunk_shape = (100, 100, 16)
    #chunk_shape = (434, 323, 251)

    porosity_points_py2hdf5(
        porosity_file_path,
        hdf5_file_path,
        hypercube_shape,
        chunk_shape,
        wells_coords,
        mult_factor,
    )
