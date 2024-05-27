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


def porosity_points_py2hdf5(porosity_file: str, por_col: str,
                            hdf5_file_path: str, config_file_path: pathlib.Path,
                            hypercube_shape: tuple, chunk_shape: tuple,
                            mult_factor: int):
    print(f"[porosity_points_py2hdf5] Expected shape: {hypercube_shape}")

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
    print(f"Target cube shape: {porosity_h5_dset.shape}")

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
    porosity_np = pd.read_csv(porosity_file_path)

    print(f"[porosity_points_py2hdf5] Loading config file")
    real_points = get_wells_coords(config_file_path)

    try:
        # Define the well id for every point
        porosity_np['well_id'] = porosity_np[['area_x', 'area_y']].apply(
            lambda row: real_points.index([row['area_x'], row['area_y']]),
            axis=1)
    except Exception as e:
        err_msg = "ERRO!: Confira se as coordenadas dos poços no arquivo de configuração"
        err_msg += " estão relativas a área alvo!"
        print(err_msg)
        raise e

    real_data_ring = 0
    porosity_np['ring'] = real_data_ring
    porosity_np['real'] = common.RealValues.real

    porosity_np = porosity_np.astype({
        'ring': np.int64,
        'real': np.int64,
        'area_x': np.int64,
        'area_y': np.int64,
        'area_z': np.int64
    })

    print(f"[porosity_points_py2hdf5] Updating {len(porosity_np)} values")

    for row in porosity_np.itertuples():
        x = row.area_x
        y = row.area_y
        z = row.area_z
        p = getattr(row, por_col)
        real = row.real
        ring = row.ring
        well_id = row.well_id
        porosity_h5_dset[x, y, z] = (x, y, z, p, real, ring, well_id)

    if mult_factor > 1:
        print(f"[porosity_points_py2hdf5] Adding extra "\
               "well points for large data")

        # Create a sample full depth well
        full_depth_well = np.empty(z_len, dtype=data_type)
        for z in range(z_len):
            full_depth_well[z] = (
                np.int64(0),
                np.int64(0),
                np.int64(z),
                abs(np.random.normal()),  # Random porosity
                common.RealValues.real,
                0,
                -1,
            )

        # Filter only real data
        expanded_real_points = real_points.copy()
        print('[porosity_points_py2hdf5] New well points:')
        for well_id in range(len(real_points), mult_factor * len(real_points)):
            # Get a new random coordinate for the well and update it in the
            # full_depth_well to be copied
            new_x, new_y = _new_random_coord(x_len, y_len, expanded_real_points)
            expanded_real_points.append((new_x, new_y))
            print(f'({new_x}, {new_y})')
            full_depth_well['x'] = new_x
            full_depth_well['y'] = new_y
            full_depth_well['well_id'] = well_id

            # Add new well to h5 dataset
            porosity_h5_dset[np.int64(new_x),
                             np.int64(new_y), :] = full_depth_well[:]

    porosity_h5_f.close()

def get_wells_coords(config_file_path: pathlib.Path) -> list:
    config = config_parser.YAMLConfig(config_file_path)
    real_points = config.wells_as_simple_list
    return real_points


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
        help="The csv file path with the wells porosities. As it may be a subset "
        "of the whole 3D cube, it is not possible to get the cube shape "
        "from this.",
    )

    parser.add_argument(
        '-p_col',
        dest='por_col',
        action='store',
        required=True,
        type=str,
        help=
        "The porosity column name to consider as the base porosity. Example: density_por",
    )

    parser.add_argument(
        '-config',
        dest='config_file_path',
        action='store',
        required=True,
        help=
        "The YAML config file path with wells coords. This is necessary to garantee"
        " the wells coords ordering.",
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

    parser = config_arg_parser()
    args = parser.parse_args()

    # Hypercube shape from feature file contains the padding for the
    # displacement. This is unnecessary here
    hypercube_shape = h5py.File(pathlib.Path(args.feat_file_path),
                                'r')['f'].shape
    disp_window = 3
    hypercube_shape = (hypercube_shape[0] - 2 * disp_window,
                       hypercube_shape[1] - 2 * disp_window,
                       hypercube_shape[2] - 2 * disp_window)
    print(f"Hypercube shape: {hypercube_shape}")
    porosity_file_path = args.porosity_file
    hdf5_file_path = args.hdf5_file
    mult_factor = int(args.mult_factor)
    por_col = args.por_col
    config_file_path = pathlib.Path(args.config_file_path)

    # chunk_shape = (100, 100, 16)
    # chunk_shape = (434, 323, 251)
    chunk_shape = (100, 100, hypercube_shape[2])
    print(f"chunk shape: {chunk_shape}")
    porosity_points_py2hdf5(
        porosity_file_path,
        por_col,
        hdf5_file_path,
        config_file_path,
        hypercube_shape,
        chunk_shape,
        mult_factor,
    )
