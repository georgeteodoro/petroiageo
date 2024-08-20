"""
This script filters the wells porosities data based on the
depth or time interval, may normalize data resolution via 
interpolation  and possibly aggregate the data based on 
some strategy.
See the -h option for more.
"""
import argparse
import pandas as pd
import pathlib
import numpy as np
from dataclasses import dataclass
from enum import Enum

REQUIRED_DEPTH_COL_AFTER_AGG = "depth"
DEFAULT_N_METERS = 1
DEFAULT_ROLLING_W = 3


class AggregationStrategy(Enum):
    NONE = "None"
    M_O_R = "mean_of_rolling"
    M_O_N = "mean_of_n_meters"

    def as_list():
        return [str(opt.name) for opt in AggregationStrategy]

    def explain_str():
        result_str = "NONE: Dont aggregate the measurements by any means."
        result_str += "\n M_O_R: mean_of_rolling. Calculate the mean of rolling window" \
                        " with the result assigned to the center of window. It uses the" \
                        " rolling_w arg."
        result_str += "\n M_O_N: mean_of_n_meters. Calculate the mean of every n" \
                        " meters with the result assigned to the center of window. It uses the" \
                        " n_meters arg."
        return result_str


@dataclass
class AggregationParams:
    agg_strat: AggregationStrategy
    rolling_window: int = DEFAULT_ROLLING_W
    n_meters: int = DEFAULT_N_METERS


def print_measurements_per_well(df: pd.DataFrame):
    print(
        f"[LOG]New measurements per well: {df[['area_x', 'area_y']].value_counts().to_dict()}"
    )


def main(aggregated_por_dfs_file_path: str, target_por_file_path: str,
         starting_pre_salt_file_path: str, min_depth: float, max_depth: float,
         no_interp: bool, by_time: bool, wells_info_path: str,
         agg_params: AggregationParams):

    print(f"[LOG]Reading file {aggregated_por_dfs_file_path}")
    por_df = pd.read_csv(aggregated_por_dfs_file_path)

    print(
        f"[LOG]Found num measures for wells: {por_df[['area_x', 'area_y']].value_counts().to_dict()}"
    )

    target_filter_col = "time" if by_time else "depth"

    print(
        f"[LOG]Filtering in {target_filter_col} interval"
    )
    final_df = filter_data(starting_pre_salt_file_path, min_depth, max_depth,
                           wells_info_path, por_df, target_filter_col)

    print_measurements_per_well(final_df)

    if no_interp:
        print("[LOG] Didn't interpolated data!")
    else:
        final_df = normalize_resolution(final_df, target_filter_col)

    final_df.sort_values(['area_x', 'area_y', target_filter_col],
                         inplace=True,
                         ascending=True)

    print_measurements_per_well(final_df)

    final_df = agg_porosities(final_df, agg_params, target_filter_col)

    final_df = final_df.round(4)

    print_measurements_per_well(final_df)

    print(f"[LOG]Saving final df at {target_por_file_path}")
    pathlib.Path(target_por_file_path).parent.mkdir(exist_ok=True, parents=True)
    final_df.to_csv(target_por_file_path, index=None)


def filter_data(starting_pre_salt_file_path: str, min_depth: float,
                max_depth: float, wells_info_path: str, por_df: pd.DataFrame,
                target_filter_col: str) -> pd.DataFrame:
    """
    Filter data depending on the target_filter_col value.
    """
    final_df = None
    if target_filter_col == 'time':
        final_df = filter_by_time(starting_pre_salt_file_path, min_depth,
                                  wells_info_path, por_df)
    else:
        final_df = por_df[(por_df[target_filter_col] >= min_depth)
                          & (por_df[target_filter_col] <= max_depth)]

    return final_df


def filter_by_time(starting_pre_salt_file_path: str, min_time: float,
                   wells_info_path: str, por_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data by time. Each Well may have it's own time start defined in 
    starting_pre_salt_file_path. Otherwise, min_time is used for every well.
    """
    wells_info_df = None
    starting_pre_salt_per_well_df = None
    if starting_pre_salt_file_path:
        starting_pre_salt_per_well_df = pd.read_csv(starting_pre_salt_file_path)
        try:
            wells_info_df = pd.read_csv(wells_info_path)
        except Exception as e:
            print(
                "[ERROR] When using --by_time and --starting_pre_salt_file_path "
                + "is defined, you should define --wells_info")
            raise e

    wells_coords = por_df[['global_x',
                           'global_y']].value_counts().index.to_list()
    final_df = None

    for global_x, global_y in wells_coords:
        target_data = por_df[(por_df['global_x'] == global_x)
                             & (por_df['global_y'] == global_y)]

        start_time_pre_salt = get_pre_salt_start(starting_pre_salt_per_well_df,
                                                 wells_info_df, global_x,
                                                 global_y, min_time)
        target_data = target_data[target_data['time'] >= start_time_pre_salt]
        time_min = target_data['time'].min()
        time_max = target_data['time'].max()
        print(f"[LOG]Data time interval: [{time_min},{time_max}]. Total time: {time_max-time_min:.3f}")
        final_df = pd.concat([final_df, target_data])
    return final_df


def get_pre_salt_start(starting_pre_salt_per_well_df: pd.DataFrame,
                       wells_info_df: pd.DataFrame, global_x: int,
                       global_y: int, min_time: float) -> float:
    time_start = None
    if starting_pre_salt_per_well_df is not None:
        well_name = wells_info_df[
            (wells_info_df['x_coord'] == global_x)
            & (wells_info_df['y_coord'] == global_y)].head(1)['Well'].item()
        print(f"[LOG] {well_name}")
        time_start = starting_pre_salt_per_well_df[
            starting_pre_salt_per_well_df['Well'] == well_name]['time'].item()
    else:
        time_start = min_time

    return time_start


def normalize_resolution(por_df: pd.DataFrame,
                         target_filter_col: str) -> pd.DataFrame:
    """
    Change the resolution of all wells based on the one with the most measurements.
    It uses an interpolation to change resolutions.
    Return a DataFrame 
    """
    wells_coords_and_counts = por_df[['area_x',
                                      'area_y']].value_counts().to_dict()
    #https://datagy.io/python-get-dictionary-key-with-max-value/
    coord_with_max_n_measures = max(wells_coords_and_counts,
                                    key=wells_coords_and_counts.get)

    print(f"[LOG]Coord with max measurements: {coord_with_max_n_measures}" +
          ". Will change the resolution of the others to match this one.")

    # At this point, the coord_with_max_n_measures has the greatest resolution
    # lets bring all of the other wells measurements to that resolution
    final_df = por_df[(por_df['area_x'] == coord_with_max_n_measures[0])
                      & (por_df['area_y'] == coord_with_max_n_measures[1])]

    target_z_measures = final_df[target_filter_col].values

    print(f"[LOG]Changing resolution")
    for well_coords in wells_coords_and_counts.keys():
        if well_coords != coord_with_max_n_measures:
            target_data = por_df[(por_df['area_x'] == well_coords[0])
                                 & (por_df['area_y'] == well_coords[1])]

            cols_with_fixed_values = [
                'area_x', 'area_y', 'global_x', 'global_y', "depth", "time"
            ]
            curr_depth_measures = target_data[target_filter_col].values
            curr_well_new_data = dict()
            for col in target_data.columns:
                if col not in cols_with_fixed_values:
                    new_col_measures = np.interp(target_z_measures,
                                                 curr_depth_measures,
                                                 target_data[col])
                    curr_well_new_data[col] = new_col_measures
                elif col not in ["depth", 'time']:
                    curr_well_new_data[col] = [
                        target_data.head(1)[col].values[0]
                    ] * len(target_z_measures)

            curr_well_new_data[target_filter_col] = target_z_measures
            final_df = pd.concat([pd.DataFrame(curr_well_new_data), final_df])
    return final_df


def agg_porosities(df: pd.DataFrame,
                   agg_params: AggregationParams,
                   target_filter_col: str = 'depth') -> pd.DataFrame:
    """
    Aggregate the df based on the mathods defined in agg_params.
    df: pd.DataFrame
    agg_params: AggregationParams
    Return:
    pd.DataFrame
    """
    if agg_params.agg_strat == AggregationStrategy.NONE:
        print(f"[LOG]DONT AGG POROSITY MEASURES")
        final_df = df
    elif agg_params.agg_strat == AggregationStrategy.M_O_R:
        final_df = aggregate(
            df, agg_wells_dfs_mean_rolling_w(agg_params.rolling_window))
    elif agg_params.agg_strat == AggregationStrategy.M_O_N:
        final_df = aggregate(
            df, agg_wells_dfs_n_meters(agg_params.n_meters, target_filter_col))
    else:
        raise ValueError(
            f"agg_strat should be one of {AggregationStrategy.as_list()}")

    return final_df


def aggregate(df, agg_func):
    final_df = None
    wells_coords = sorted(df[['area_x',
                              'area_y']].value_counts().index.to_list())
    for well_coord in wells_coords:
        target_data = df[(df['area_x'] == well_coord[0])
                         & (df['area_y'] == well_coord[1])].copy()

        grouped = agg_func(target_data)

        if final_df is None:
            final_df = grouped
        else:
            final_df = pd.concat([final_df, grouped])

    return final_df


def agg_wells_dfs_n_meters(n_meters: int, target_filter_col: str):
    """
    Returns a function that aggregates the porosities every n meters using the mean.
    
    """
    print(f"[LOG]AGG DFS EVERY {n_meters} {target_filter_col.upper()}S")

    def agg_func(df: pd.DataFrame) -> pd.DataFrame:
        df['group_indicator'] = df[target_filter_col] // n_meters

        grouped = df.groupby(by='group_indicator').mean()
        grouped = grouped.reset_index()
        grouped.drop("group_indicator", inplace=True, axis=1)

        return grouped

    return agg_func


def agg_wells_dfs_mean_rolling_w(rolling_w: int):
    """
    Returns a function that applies a rolling window with mean func to the 
    data of every well.
    """
    print(f"[LOG]AGG DFS MEAN ROLLING WINDOW {rolling_w}")

    def agg_func(df: pd.DataFrame) -> pd.DataFrame:
        curr_df = df.rolling(rolling_w, min_periods=1, center=True).mean()
        return curr_df

    return agg_func


def check_for_int_depth_measures(depth_values: np.ndarray):
    min_z = np.min(depth_values)
    max_z = np.max(depth_values)
    expected_depth_array = np.arange(min_z, max_z + 1, 1)
    assert np.array_equal(
        expected_depth_array, depth_values
    ), f"Expected array: \n {expected_depth_array}\n True array: {depth_values}"


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="This script filters the wells porosities data based on the \
        depth interval, normalize data resolution via interpolation \
        and possibly aggregate the data based on some strategy.")

    parser.add_argument(
        "--min_depth",
        required=True,
        type=float,
        help=
        "The min depth used to filter out the common depth interval of wells.")

    parser.add_argument(
        "--max_depth",
        required=True,
        type=float,
        help=
        "The max depth used to filter out the common depth interval of wells.")

    parser.add_argument(
        "--por_file",
        required=True,
        type=str,
        help="The target aggregated wells porosities file. Must be a csv")

    parser.add_argument("--target_por_file",
                        required=True,
                        type=str,
                        help="The target normalized aggregated porosity file")

    parser.add_argument(
        '--agg_strat',
        required=False,
        type=str,
        choices=AggregationStrategy.as_list(),
        default=AggregationStrategy.NONE.name,
        help=f"The strategy for aggregating porosities measures." +
        f" Options: {AggregationStrategy.explain_str()}." +
        f" Default: {AggregationStrategy.NONE.name}")

    parser.add_argument(
        "--rolling_w",
        required=False,
        type=int,
        default=DEFAULT_ROLLING_W,
        help="The window used for the rolling window aggregation methods." +
        f"Type: integer. Default: {DEFAULT_ROLLING_W}")

    parser.add_argument(
        "--n_meters",
        required=False,
        type=int,
        default=DEFAULT_N_METERS,
        help=
        "The amount of meters used to take the mean. when aggregating with M_O_N"
        + f" Type: integer. Default: {DEFAULT_N_METERS}")

    parser.add_argument(
        '--no_interp',
        action='store_true',
        required=False,
        help="Flag that indicates we shouldn't interpolate the data")

    parser.add_argument(
        '--by_time',
        action='store_true',
        required=False,
        help="Flag that indicates we should operate on the time column " +
        "instead of depth column")

    parser.add_argument(
        "--start_time_pre_salt_path",
        required=False,
        type=str,
        default=None,
        help="The csv file indicating for every well the pre salt start " +
        "time. It should have two columns: Well and time. This might be" +
        " used when using --by_time. If not defined, use the " +
        "--min_depth value for every well.")

    parser.add_argument(
        "--wells_info",
        required=False,
        type=str,
        default=None,
        help="The file with wells info. This is needed when using " +
        "--by_time and --start_time_pre_salt_path is defined.")

    return parser


if __name__ == "__main__":
    args = config_parser().parse_args()

    agg_params = AggregationParams(AggregationStrategy[args.agg_strat],
                                   args.rolling_w, args.n_meters)

    main(args.por_file, args.target_por_file, args.start_time_pre_salt_path,
         args.min_depth, args.max_depth, args.no_interp, args.by_time,
         args.wells_info, agg_params)
