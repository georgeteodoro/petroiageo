"""
This script generates the main input seismic/porosity cube to be used by the algorithm.
It requires the wells coordinates information along with the wells porosity file
"""
import argparse
import h5py
from math import prod
import numpy as np
import pathlib
from tqdm import tqdm

import sys
sys.path.insert(0,'../alg/')

from alg import common

def porosity_points_py2hdf5(porosity_file_path, hdf5_file_path, hypercube_shape, chunk_shape,
                            real_points):
    porosity_np = np.load(pathlib.Path(porosity_file_path))
    (x, y, z) = hypercube_shape

    print('[porosity_points_py2hdf5] Creating hdf5 file')
    new_h5_porosity_path = pathlib.Path(hdf5_file_path)
    porosity_h5_f = h5py.File(new_h5_porosity_path, 'w')
    porosity_h5_dset = porosity_h5_f.create_dataset(
        'p',
        hypercube_shape,
        dtype=np.dtype([('x', np.int64), ('y', np.int64), ('z', np.int64),
                        ('phi', np.float64), ('real', np.int64),
                        ('well_id', np.int64)]),
        chunks=chunk_shape,
    )

    print(f'[porosity_points_py2hdf5] Filling coordinates')
    for i in tqdm(range(x)):
        # Batching of yz coordinates for writing on hdf5 file
        all_yz = []
        for j in range(y):
            # Batching of z coordinates for writing on hdf5 file
            all_z = []
            for k in range(z):
                all_z = all_z + [(i, j, k, 0, common.RealValues.empty, -1)]
            all_yz = all_yz + [all_z]
        # Commit all points for a given x coordinate
        porosity_h5_dset[i, ...] = all_yz

    print(f'[porosity_points_py2hdf5] Updating {len(porosity_np)} values')
    for (x, y, z, p) in tqdm(porosity_np):
        if (x, y) in real_points:
            real = common.RealValues.real
            well_id = real_points.index((x, y))
        else:
            real = common.RealValues.canal
            well_id = -1
        porosity_h5_dset[np.int64(x), np.int64(y),
                         np.int64(z)] = (np.int64(x), np.int64(y), np.int64(z),
                                         p, real, well_id)

    porosity_h5_f.close()

def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='POV')

    parser.add_argument('--base-feat-file',
                        dest='feat_file_path',
                        action='store',
                        required=True,
                        help="The base npy feat file path to get the hypercube shape from")
    parser.add_argument('--porosity-file-path',
                        dest='porosity_file',
                        action='store',
                        required=True,
                        help='The file path with the base porosity. As it may be a subset of the whole 3D cube,\
                        it is not possible to get the cube shape from this.')

    parser.add_argument('--hdf5-file-path',
                        dest='hdf5_file',
                        action='store',
                        required=True,
                        help='The hdf5 file path to be generated.')
if __name__ == '__main__':
    wells_coords = [(134, 227), (146, 500), (167, 186), (174, 365), (200, 102),
                  (236, 113), (250, 315), (287, 242), (230, 194), (344, 276)]
    
    parser = config_arg_parser()
    args = parser.parse_args()
    hypercube_shape = np.load(pathlib.Path(args.base_feat_file)).shape
    porosity_file_path = args.porosity_file
    hdf5_file_path = args.hdf5_file
    chunk_shape = (100, 100, 251)

    porosity_points_py2hdf5(porosity_file_path, hdf5_file_path, hypercube_shape,
                            chunk_shape, wells_coords)