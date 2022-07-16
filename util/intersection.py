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
    new_arg_parser.add_argument("--second", type=str, required=True)

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
    
    first_df = pd.read_csv(first_file)
    second_df = pd.read_csv(second_file)

    columns_of_merge = ['x','y','z']
    intersection = pd.merge(first_df, second_df, how='inner', on=columns_of_merge)

    columns_of_interest = ['x','y','z','phi']
    intersection = intersection.loc[:, columns_of_interest]

    print(f"intersection size: {len(intersection)}")

    print(intersection.head())

    INTERSECTION_FILE_PATH = pathlib.Path("./intersection.csv")
    intersection.to_csv(INTERSECTION_FILE_PATH, index=None)