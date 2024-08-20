"""
This script interpolates the required seismic data for a given area
based on the area limits and the depths of the area's well's porosities
 measurements. See the -h option for more.
"""
import argparse
import math
import numpy as np
import pandas as pd
import pathlib
from timeit import default_timer as timer


def main(por_file_path: str, area_info_file_path: str, target_area: int,
         seismic_file_path: str, seismic_resolution: float,
         seismic_start_depth: float, target_seismic_file_path: str,
         target_merge_file_path: str, dont_interpolate: bool, by_time: bool):

    print(
        f"[LOG]Loading target area {target_area} info from {area_info_file_path}"
    )
    areas_info = pd.read_csv(area_info_file_path)
    target_area_info = areas_info[areas_info['area'] == target_area]
    print(f"[LOG]Target area info:\n{target_area_info}")

    target_z_col = 'time' if by_time else 'depth'
    print(f"[LOG]Z dimension is  {target_z_col}")

    print(f"[LOG]Loading wells data from {por_file_path}")
    wells_data = pd.read_csv(por_file_path)
    wells_data.sort_values(by=['area_x', 'area_y', target_z_col],
                           inplace=True,
                           ascending=True)

    # These z_values aren't relative to the target_z_col interval, that is,
    # they're the true values
    z_values = wells_data[target_z_col].unique()
    z_values = np.sort(z_values)
    print(f"Wells {target_z_col}s:\n{z_values}")
    print(f"Wells {target_z_col}s shape: {z_values.shape}")

    # The wells z_values are already treated so this is ok
    min_z = z_values.min()
    max_z = z_values.max()

    z_interval = (min_z, max_z)
    x_interval = (target_area_info['min_x'].item(),
                  target_area_info['max_x'].item())
    y_interval = (target_area_info['min_y'].item(),
                  target_area_info['max_y'].item())
    print(
        f"[LOG]Target intervals: x: {x_interval}, y: {y_interval}, z: {z_interval}"
    )

    print(
        f"[LOG]Loading seismic data from {seismic_file_path} and filtering based on target intervals"
    )
    z_filtered_seismic_data, z_idx_interval = filter_seismic_and_get_interval(
        seismic_file_path, seismic_start_depth, seismic_resolution, z_interval,
        x_interval, y_interval)

    print(f"[LOG]Filtered seismic data shape:\n{z_filtered_seismic_data.shape}")

    seismic_z_interval = list(
        range(z_idx_interval[0] * seismic_resolution,
              z_idx_interval[1] * seismic_resolution + 1, seismic_resolution))

    print(f"[LOG]Seismic z interval:\n{seismic_z_interval}")
    print(f"[LOG]Seismic z interval len:\n{len(seismic_z_interval)}")

    if not dont_interpolate:
        print(f"[LOG]Interpolating seismic data")
        start_time = timer()
        interp_data = interpolate_data(z_values, seismic_z_interval,
                                       z_filtered_seismic_data)
        end_time = timer()
        # Clear memory
        z_filtered_seismic_data = None
        print(f"[LOG]Interpolation done in {end_time-start_time:.2f} seconds")
        print(f"[LOG]Interpolated seismic data shape: {interp_data.shape}")
    else:
        print(f"[LOG] Didn't interpolate data!")
        interp_data = z_filtered_seismic_data

    print(f"[LOG]Saving seismic data to {target_seismic_file_path}")
    pathlib.Path(target_seismic_file_path).parent.mkdir(exist_ok=True,
                                                        parents=True)
    np.save(target_seismic_file_path, interp_data)

    merge_well_with_seismic(target_merge_file_path, target_z_col, wells_data,
                            min_z, interp_data, seismic_resolution)


def merge_well_with_seismic(target_merge_file_path: str, target_z_col: str,
                            wells_data: pd.DataFrame, min_z: float,
                            seismic_data: np.ndarray, seismic_res: int):
    print(f"[LOG]Merging wells data with its seismic data")
    unique_area_wells_coords = sorted(
        list(wells_data[['area_x', 'area_y']].value_counts().index))
    # Assure the coords are integers
    unique_area_wells_coords = [(int(c[0]), int(c[1]))
                                for c in unique_area_wells_coords]
    print(f"[LOG]Unique wells coords found: {unique_area_wells_coords}")
    wells_seismic_values = np.empty(0)
    for well_coord in unique_area_wells_coords:
        well_seismic = seismic_data[well_coord[0], well_coord[1], :]
        # On the case that we have "broken" data. That is
        # wells porosity target_z_col aren't perfectly contiguous
        # We subtract min_z because the seismic_data is already z filtered
        # So we must fix the well_coord's[target_z_col] column values
        well_zs = ((wells_data[
            (wells_data['area_x'] == well_coord[0])
            & (wells_data['area_y'] == well_coord[1])][target_z_col] -
                   min_z) // seismic_res).astype(int)
        well_seismic = well_seismic[well_zs]
        wells_seismic_values = np.concatenate(
            [wells_seismic_values, well_seismic])

    # Clear memory
    seismic_data = None
    wells_data['seismic'] = wells_seismic_values
    wells_data[f'area_{target_z_col}'] = wells_data[target_z_col] - min_z

    print(f"[LOG]Saving merged wells data at {target_merge_file_path}")
    pathlib.Path(target_merge_file_path).parent.mkdir(exist_ok=True,
                                                      parents=True)
    wells_data.to_csv(target_merge_file_path, index=None)


def interpolate_data(target_depths: np.ndarray, curr_depths: np.ndarray,
                     data: np.ndarray) -> np.ndarray:
    """
    Interpolate the seismic data on the z/depth/time axis (third one).
    target_depths: The target points at where we want to calc new values
    curr_depths: The current depths for the data z axis
    data: The data to interpolate
    Return the interpolated data
    """
    interp_data = np.empty((data.shape[0], data.shape[1], target_depths.size))
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            curr_interp = np.interp(target_depths, curr_depths, data[x, y, :])
            interp_data[x, y, :] = curr_interp

    return interp_data


def filter_seismic_and_get_interval(seismic_path: str,
                                    seismic_start_depth: float,
                                    resolution: float, target_z_interval: tuple,
                                    target_x_interval: tuple,
                                    target_y_interval: tuple) -> tuple:
    """
    Load and filter the seismic data based on:
    
    seismic_path: The seismic data npy path
    seismic_start_depth: The starting z depth of the seismic data
    resolution: Seismic data z axis resolution in meters or time
    target_z_interval: A tuple of (min_z, max_z)
    target_x_interval: A tuple of (min_x, max_x)
    target_y_interval: A tuple of (min_y, max_y)

    Returns a tuple of: (filtered data, z idx interval)
    where z idx interval is a tuple of (min_z_idx, max_z_idx)
    """
    seismic_data = np.load(seismic_path)
    seismic_z_count = seismic_data.shape[-1]

    min_z, max_z = target_z_interval
    min_x, max_x = target_x_interval
    min_y, max_y = target_y_interval

    target_z_min_seismic_idx = int(max((min_z - seismic_start_depth) // resolution,
                                   0))
    target_z_max_seismic_idx = min(
        math.ceil((max_z - seismic_start_depth) / resolution),
        seismic_z_count - 1)
    
    z_filtered_seismic_data = seismic_data[
        min_x:max_x + 1, min_y:max_y + 1,
        target_z_min_seismic_idx:target_z_max_seismic_idx + 1]

    z_idx_interval = (target_z_min_seismic_idx, target_z_max_seismic_idx)
    return z_filtered_seismic_data, z_idx_interval


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "This script interpolates the required seismic data for a given area \
        based on the area limits and the depths of the area's well's porosities \
        measurements.")

    parser.add_argument(
        "--por_file",
        required=True,
        type=str,
        help="The target treated wells porosities file. Must be a csv")

    parser.add_argument("--area_info_file",
                        required=True,
                        type=str,
                        help="The target areas info file. Must be a csv")

    parser.add_argument(
        "--target_area",
        required=True,
        type=int,
        help="The target area from where we will get the target egdes.")

    parser.add_argument("--seismic_file",
                        required=True,
                        type=str,
                        help="The original seismic file. Must be a .npy")

    parser.add_argument(
        "--seismic_resolution",
        required=True,
        type=int,
        help="The seismic file z resolution. Must be an integer")

    parser.add_argument("--start_seismic_depth",
                        required=False,
                        default=0,
                        type=float,
                        help="The seismic data starting depth.")

    parser.add_argument(
        "--target_seismic_file",
        required=True,
        type=str,
        help="The filtered and treated seismic file target path. Must be a .npy"
    )

    parser.add_argument(
        "--target_merge_file",
        required=True,
        type=str,
        help="The target file that will contain the original por file with" +
        " its respective seismic values. Must be a csv")

    parser.add_argument(
        '--no_interp',
        action='store_true',
        required=False,
        help="Flag that indicates we shouldn't interpolate the data")

    parser.add_argument(
        '--by_time',
        action='store_true',
        required=False,
        help="Flag indicating the seismic file z dimension is in time and so" +
        " we should use the time column of por_file.")

    return parser


if __name__ == "__main__":
    args = config_parser().parse_args()

    main(args.por_file, args.area_info_file, args.target_area,
         args.seismic_file, args.seismic_resolution, args.start_seismic_depth,
         args.target_seismic_file, args.target_merge_file, args.no_interp,
         args.by_time)
