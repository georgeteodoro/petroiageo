"""
This program saves the intersection of two csv files by the x,y and z columns.

The final csv generated should be able to have the x,y,z and phi columns
"""
import pathlib
import pandas as pd
import argparse
import sys


def config_arg_parser():
    new_arg_parser = argparse.ArgumentParser()
    new_arg_parser.add_argument("--first", type=str, required=True)
    new_arg_parser.add_argument("--first-sep", type=str, default=",")
    new_arg_parser.add_argument("--second", type=str, required=True)
    new_arg_parser.add_argument("--second-sep", type=str, default=",")
    new_arg_parser.add_argument(
        "--phi-of",
        type=str,
        choices=["first", "second"],
        required=True,
        help="What phi to save: first or second",
    )
    new_arg_parser.add_argument(
        "--it",
        type=int,
        default=None,
        help="A suffix number to be used in the final intersection file name",
    )
    return new_arg_parser


if __name__ == "__main__":
    arg_parser = config_arg_parser()

    args = arg_parser.parse_args()

    first_file = pathlib.Path(args.first)
    second_file = pathlib.Path(args.second)

    if any([not first_file.exists(), not second_file.exists()]):
        print("Some of the paths doesn't exists!")
        sys.exit(-1)

    if any([not first_file.is_file(), not second_file.is_file()]):
        print("Some of the paths isn't a file!")
        sys.exit(-1)

    if args.second_sep == "\\s":
        args.second_sep = " "

    if args.first_sep == "\\s":
        args.first_sep = " "

    first_df = pd.read_csv(first_file, sep=args.first_sep)
    second_df = pd.read_csv(second_file, sep=args.second_sep)

    columns_of_merge = ["x", "y", "z"]
    intersection = pd.merge(
        first_df, second_df, how="inner", on=columns_of_merge
    )

    columns_of_interest = ["x", "y", "z"]

    if "phi" in intersection.columns:
        columns_of_interest.append("phi")

    else:
        # If phi_x is present it means that phi_y is also present
        if "phi_x" in intersection.columns:
            if args.phi_of == "first":
                columns_of_interest.append("phi_x")
            else:
                columns_of_interest.append("phi_y")

    intersection = intersection.loc[:, columns_of_interest]

    print(f"intersection size: {len(intersection)}")

    print(intersection.head())

    if not args.it == None:
        intersection_file_name = f"intersection{args.it}.csv"
    else:
        intersection_file_name = "intersection.csv"

    INTERSECTION_FILE_PATH = pathlib.Path(f"./{intersection_file_name}")
    intersection.to_csv(INTERSECTION_FILE_PATH, index=None)
