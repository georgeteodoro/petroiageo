import numpy as np
from math import prod
import h5py
from tqdm import tqdm

import common


def porosity_points_py2hdf5(filename, hypercube_shape, chunk_shape,
                            real_points):
    porosity_np = np.load(filename)
    (x, y, z) = hypercube_shape

    print('[porosity_points_py2hdf5] Creating hdf5 file')
    porosity_h5_f = h5py.File(f'./dados/porosity_data.h5', 'w')
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
                all_z = all_z + [(i, j, k, 0, common.RealValues.empty, 0)]
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


if __name__ == '__main__':
    real_wells = [(134, 227), (146, 500), (167, 186), (174, 365), (200, 102),
                  (236, 113), (250, 315), (287, 242), (230, 194), (344, 276)]
    hypercube_shape = np.load(f'./dados/NEAR.npy').shape

    porosity_points_py2hdf5('./dados/porosity-canal.npy', hypercube_shape,
                            (100, 100, 100), real_wells)
