"""
The goal of this script is to zero out porosities measures through
x and y coords of selected wells of a cube inside a h5 file.
The other wells points are fixed so that their well_id correspond
to their new well_id.
Example:
wells:
    coords:
        - [1,1]
        - [2,2]
        - [3,3]

--wells_idxs 2

Than all points belonging to the well [3,3] should now have well_id == 1

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

    z_shape = data.shape[2]
    # MODIFIES THE INPUT FILE INPLACE
    new_well_id = 0
    for original_well_id, (x,y) in enumerate(config.wells_as_simple_list):
        print(f"Curr well: Orig id: {original_well_id}, coords: {x}, {y}")
        if original_well_id in args.wells_idxs:
            print(f"Zeroing it out!")
            all_z = [() for _ in range(z_shape)]
            for z in range(z_shape):
                all_z[z] = (x, y, z, 0, common.RealValues.empty, -1, -1)
            data[x, y, ...] = all_z    
        else:
            print(f"It is not on wells idxs!")
            if new_well_id != original_well_id:
                print(f"Should update its well id!")
                original_data = data[x,y]
                original_data['well_id'] = new_well_id
                data[x,y] = original_data
                print(f"NEW DATA:\n {data[x,y]}")
            else:
                print(f"As its original id is iqual to the new id ({new_well_id}) "+
                      "we dont do anything")
        
        new_well_id+=1