"""
The goal of this script is to zero out porosities measures through
x and y coords of selected wells of a cube inside a h5 file.

IMPORTANT: THIS SCRIPT MODIFIES THE INPUT H5 FILE INPLACE.
SO YOU SHOULD PASS A COPY OF THE ORIGINAL FILE AS INPUT SO THIS
MAY MODIFY IT WITHOUT LOSING THE ORIGINAL FILE
"""
import argparse
import pathlib
import h5py
import numpy as np
import sys

sys.path.insert(0, "..")

from alg import common, config_parser

def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
        help="The hdf5 file path to be modified.",
    )

    parser.add_argument(
        '--config',
        required=True,
        help="The config file path where to read the wells coords from.",
    )

    parser.add_argument('--wells_idxs',
                        type=int,
                        nargs="+",
                        required=True,
                        help="The wells indexes in config file to zero out")
    

    return parser

if __name__ == "__main__":
    parser = config_arg_parser()
    args = parser.parse_args()

    config = config_parser.YAMLConfig(pathlib.Path(args.config))
    data = h5py.File(pathlib.Path(args.input), 'r+')[common.POROSITY_DSET_NAME]
    print(f"Data shape: {data.shape}")

    # MODIFIES THE INPUPT FILE INPLACE
    for (x,y) in config.get_coords_of_target_wells_ids(args.wells_idxs):
        data[x,y,:] = (0, common.RealValues.empty, -1, -1)