"""
This script filters the wells porosities data based on the
depth interval, normalize data resolution via interpolation 
and possibly aggregate the data based on some strategy.
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
    M_O_M = "mean_of_meter"
    NONE = "None"
    M_O_R = "mean_of_rolling"
    M_O_N = "mean_of_n_meters"

    def as_list():
        return [str(opt.name) for opt in AggregationStrategy]

    def explain_str():
        result_str = "M_O_M: mean_of_meter. For every integer meter of measure, agg " \
                        "all measurements inside that meter by the mean."
        result_str += "\n NONE: Dont aggregate the measurements by any means."
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
         min_depth: float, max_depth: float, agg_params: AggregationParams):

    print(f"[LOG]Reading file {aggregated_por_dfs_file_path}")
    por_df = pd.read_csv(aggregated_por_dfs_file_path)

    print(
        f"[LOG]Found num measures for wells: {por_df[['area_x', 'area_y']].value_counts().to_dict()}"
    )

    print(f"[LOG]Filtering in depth interval [{min_depth}, {max_depth}]")
    final_df = por_df[(por_df['z'] >= min_depth) & (por_df['z'] <= max_depth)]

    print_measurements_per_well(final_df)

    final_df = normalize_resolution(final_df)
    final_df.sort_values(['area_x', 'area_y', 'z'],
                         inplace=True,
                         ascending=True)

    print_measurements_per_well(final_df)

    final_df = agg_porosities(final_df, agg_params)

    final_df = final_df.round(4)

    print_measurements_per_well(final_df)

    print(f"[LOG]Saving final df at {target_por_file_path}")
    pathlib.Path(target_por_file_path).parent.mkdir(exist_ok=True, parents=True)
    final_df.to_csv(target_por_file_path, index=None)


def normalize_resolution(por_df: pd.DataFrame) -> pd.DataFrame:
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

    target_z_measures = final_df['z'].values

    print(f"[LOG]Changing resolution")
    for well_coords in wells_coords_and_counts.keys():
        if well_coords != coord_with_max_n_measures:
            target_data = por_df[(por_df['area_x'] == well_coords[0])
                                 & (por_df['area_y'] == well_coords[1])]

            cols_with_fixed_values = [
                'area_x', 'area_y', 'global_x', 'global_y', 'z'
            ]
            curr_depth_measures = target_data['z'].values
            curr_well_new_data = dict()
            for col in target_data.columns:
                if col not in cols_with_fixed_values:
                    new_col_measures = np.interp(target_z_measures,
                                                 curr_depth_measures,
                                                 target_data[col])
                    curr_well_new_data[col] = new_col_measures
                elif col != "z":
                    curr_well_new_data[col] = [
                        target_data.head(1)[col].values[0]
                    ] * len(target_z_measures)

            curr_well_new_data['z'] = target_z_measures
            final_df = pd.concat([pd.DataFrame(curr_well_new_data), final_df])
    return final_df


def agg_porosities(porosity_dfs_dict: pd.DataFrame,
                   agg_params: AggregationParams) -> pd.DataFrame:
    if agg_params.agg_strat == AggregationStrategy.M_O_M:
        final_df = agg_wells_dfs_meter_by_meter(porosity_dfs_dict)
    elif agg_params.agg_strat == AggregationStrategy.NONE:
        print(f"[LOG]DONT AGG POROSITY MEASURES")
        final_df = porosity_dfs_dict
    elif agg_params.agg_strat == AggregationStrategy.M_O_R:
        final_df = agg_wells_dfs_mean_rolling_w(porosity_dfs_dict,
                                                agg_params.rolling_window)
    else:
        raise ValueError(
            f"agg_strat should be one of {AggregationStrategy.as_list()}")

    return final_df


def agg_wells_dfs_mean_rolling_w(por_df: pd.DataFrame,
                                 rolling_w: int) -> pd.DataFrame:
    """
    Apply a rolling window with mean func to the data of every well.
    Return a DataFrame
    """
    print(f"[LOG]AGG DFS MEAN ROLLING WINDOW {rolling_w}")
    final_df = None
    wells_coords = sorted(por_df[['area_x',
                                  'area_y']].value_counts().index.to_list())
    for well_coord in wells_coords:
        target_data = por_df[(por_df['area_x'] == well_coord[0])
                             & (por_df['area_y'] == well_coord[1])]
        curr_df = target_data.rolling(rolling_w, min_periods=1,
                                      center=True).mean()
        if final_df is None:
            final_df = curr_df
        else:
            final_df = pd.concat([final_df, curr_df])

    return final_df


def agg_wells_dfs_meter_by_meter(por_df: pd.DataFrame) -> pd.DataFrame:
    """
    The porosities measures have resolution below one meter. We aggregate 
    the porosities by meter using the mean.

    Returns a DataFrame
    """
    print(f"[LOG]AGG DFS PER DEPTH")

    final_df = None
    wells_coords = sorted(por_df[['area_x',
                                  'area_y']].value_counts().index.to_list())
    for well_coord in wells_coords:
        target_data = por_df[(por_df['area_x'] == well_coord[0])
                             & (por_df['area_y'] == well_coord[1])].copy()

        target_data['z'] = target_data['z'].apply(
            lambda x: int(str(x).split('.')[0]))

        grouped = target_data.groupby(by='z').mean()
        grouped = grouped.reset_index()

        check_for_int_depth_measures(grouped['z'].values)

        if final_df is None:
            final_df = grouped
        else:
            final_df = pd.concat([final_df, grouped])

    return final_df


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

    return parser


if __name__ == "__main__":
    args = config_parser().parse_args()

    agg_params = AggregationParams(AggregationStrategy[args.agg_strat],
                                   args.rolling_w, args.n_meters)

    main(args.por_file, args.target_por_file, args.min_depth, args.max_depth,
         agg_params)
