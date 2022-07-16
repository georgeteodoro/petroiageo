import sys
import pandas as pd
import argparse
import pathlib

def config_args_parser():
    new_arg_parser = argparse.ArgumentParser()
    new_arg_parser.add_argument("--file-to-sort", type=str, required=True,
                        help="The unsorted predictions file path")
                        
    DEFAULT_SORTED_FOLDER = "./sorted_files"
    target_folder_help_msg = "The folder to which the sorted files should be saved"
    target_folder_help_msg += f". Default: {DEFAULT_SORTED_FOLDER}"
    new_arg_parser.add_argument("--target-folder", type=str, default=DEFAULT_SORTED_FOLDER,
                    help=target_folder_help_msg)
    return new_arg_parser


if __name__ == "__main__":
    arg_parser = config_args_parser()

    user_args = arg_parser.parse_args()

    unsorted_file = pathlib.Path(user_args.file_to_sort)

    if not unsorted_file.exists():
        print(f"{user_args.file_to_sort} não existe!")
        sys.exit(-1)
    
    dest_folder = pathlib.Path(user_args.target_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    predValues = pd.read_csv(unsorted_file, header=0)
    predValues.sort_values(by=['x','y','z'], inplace=True)
    print("Ordenou")
    sortedFileName = dest_folder / f"{unsorted_file.stem}_sorted.csv"
    print(f"Salvando em: {sortedFileName}.")
    predValues.to_csv(sortedFileName, index=None)