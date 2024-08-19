"""
This script aggregate multiple wells porosities files
for every delimited area into one file for every area.
See the -h option for more.
"""
import argparse
import pandas as pd
import pathlib


def read_wells_porosities(
        target_wells_names: list,
        wells_porosity_folder: pathlib.Path) -> dict[str, pd.DataFrame]:
    """
    Tries to find the wells porosities files at wells_porosity_folder based on 
    the wells names in target_wells_names and load then as DataFrames.
    
    Returns a dict with well_name as the key and the corresponding porosity DataFrame
    as value
    """
    all_porosities = wells_porosity_folder.glob("*")

    print(f"[LOG]TARGET WELLS NAMES: {target_wells_names}")

    target_files = dict()
    for porosity_file in all_porosities:
        well_name = porosity_file.stem
        if well_name in target_wells_names:
            target_files[well_name] = porosity_file.resolve()

    wells_found = list(target_files.keys())
    print(
        f"[LOG]FOUND POR FILES FOR {len(wells_found)} of {len(target_wells_names)} WELLS: {wells_found}"
    )

    porosity_dfs_dict = dict()
    column_names = ['depth', 'neutron_por', 'density_por', 'sonic_por']
    for well_name, por_path in target_files.items():
        por_df = pd.read_csv(por_path,
                             header=0,
                             names=column_names)
        porosity_dfs_dict[well_name] = por_df

    return porosity_dfs_dict


def main(target_area, area_coords_path: str, wells_info_path: str,
         wells_por_dir: str, target_final_df_dir_path_str: str):

    target_areas_data = get_target_areas_data(target_area, area_coords_path)
    print(f"[LOG] AREAS DATA: {target_areas_data}")

    final_df_file_name_fmt_str = 'area_{}_porosity_wells.csv'
    for area_data in target_areas_data:
        curr_area = area_data['area']
        print(f"[LOG]CURRENT AREA: {curr_area}")

        area_wells_info = get_target_area_wells_info(area_data,
                                                     wells_info_path)
        print(f"[LOG]TARGET AREA WELLS INFO:\n{area_wells_info}")

        target_wells_names = list(set(area_wells_info['Well'].tolist()))
        porosity_dfs_dict = read_wells_porosities(target_wells_names,
                                                  pathlib.Path(wells_por_dir))

        print(f"[LOG]CONCATENATING MULTIPLE DFS INTO ONE")
        final_df = None
        for well_name, well_df in porosity_dfs_dict.items():
            num_meds = len(well_df)
            well_info = area_wells_info[area_wells_info['Well'] ==
                                        well_name].iloc[0].to_dict()
            well_df['global_x'] = [well_info['x_coord']] * num_meds
            well_df['global_y'] = [well_info['y_coord']] * num_meds
            well_df['area_x'] = [well_info['x_coord'] - area_data['min_x']
                                 ] * num_meds
            well_df['area_y'] = [well_info['y_coord'] - area_data['min_y']
                                 ] * num_meds

            well_df.rename({"depth": 'z'}, axis=1, inplace=True)
            if final_df is None:
                final_df = well_df
            else:
                final_df = pd.concat([final_df, well_df])

        if target_final_df_dir_path_str[-1] != "/":
            target_final_df_dir_path_str += "/"

        target_final_df_dir_path = pathlib.Path(target_final_df_dir_path_str)
        target_final_df_dir_path.mkdir(parents=True, exist_ok=True)

        target_concated_agg_dfs_path = target_final_df_dir_path / final_df_file_name_fmt_str.format(
            curr_area)
        print(f"[LOG]SAVING FINAL DF TO: {target_concated_agg_dfs_path}")

        final_df.sort_values(['area_x', 'area_y', 'z'],
                             inplace=True,
                             ascending=True)
        final_df.reset_index(inplace=True)
        final_df.drop('index', axis=1, inplace=True)
        final_df.to_csv(target_concated_agg_dfs_path, index=False)
        print("")

    print("[LOG]Now, you should treat the porosity data for every area")


def get_target_areas_data(target_area: int,
                          area_coords_path: str) -> list[dict]:
    """
    Compile a list of areas data as dicts based on the target area.
    If the target area is negative, we get data from all areas
    """
    areas_df = pd.read_csv(pathlib.Path(area_coords_path))

    target_areas_data = list()
    if target_area >= 0:
        try:
            target_areas_data = [
                areas_df[areas_df['area'] == target_area].iloc[0].to_dict()
            ]
        except Exception as e:
            print(
                f"[ERROR]Probably, the target area {target_area} doesnt exists!"
                + f" Accepted areas: {areas_df['area'].to_list()}")
            raise e
    else:
        target_areas_data = [
            area_data.to_dict() for _, area_data in areas_df.iterrows()
        ]

    return target_areas_data


def get_target_area_wells_info(area_limits: dict,
                               wells_info_path: str) -> pd.DataFrame:
    """
    Get the data from every well inside the area limits.
    """
    all_wells_info = pd.read_csv(pathlib.Path(wells_info_path))
    area_wells_info = all_wells_info[
        (area_limits['min_y'] <= all_wells_info['y_coord'])
        & (area_limits['max_y'] >= all_wells_info['y_coord']) &
        (area_limits['min_x'] <= all_wells_info['x_coord']) &
        (area_limits['max_x'] >= all_wells_info['x_coord'])]

    return area_wells_info


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="This script aggregate multiple wells porosities files \
        for every delimited area into one file for every area.")

    parser.add_argument(
        '--target_area',
        required=True,
        type=int,
        help="The target area to filter the wells porosities. If negative, " +
        "proccess every area. Type: Integer")

    parser.add_argument('--areas_coords_path',
                        required=False,
                        type=str,
                        default="../../common_data/ANP/areas_coords.csv",
                        help="The path to the file that have the areas coords")

    parser.add_argument(
        '--wells_info_path',
        required=False,
        type=str,
        default="../../common_data/ANP/wells_info_completed.csv",
        help="The path to the file that have wells infos")

    parser.add_argument(
        '--wells_por_dir_path',
        required=False,
        type=str,
        default="../../common_data/ANP/porosity/raw/",
        help="The path to the dir that have wells porosities files. " +
        "The file names must be equal to uppercasewellname.txt")

    parser.add_argument(
        '--final_df_dir_path',
        required=False,
        type=str,
        default="../../common_data/ANP/porosity/",
        help="The path to the dir that will have the final dfs")

    return parser


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    area_coords_path = args.areas_coords_path
    wells_info_path = args.wells_info_path
    wells_por_dir = args.wells_por_dir_path
    target_final_df_dir_path = args.final_df_dir_path

    main(args.target_area, area_coords_path, wells_info_path, wells_por_dir,
         target_final_df_dir_path)
