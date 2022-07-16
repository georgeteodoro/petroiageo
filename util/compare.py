"""
This program compares two predictions.

If the two files have different sizes, it will save an equal.csv file and a difference.csv file.

It will also save a merged.csv file with rows merged by x,y and z.

It will also raise some usefull warnings.
"""
import pandas as pd
import numpy as np
import argparse
import pathlib
import math

def treat_predicted(predicted:pd.DataFrame, file_path:pathlib.Path) -> pd.DataFrame:
    COLUMNS_OF_INTEREST = ['x','y','z','phi']
    predicted = predicted.loc[:, COLUMNS_OF_INTEREST]
    predicted.set_index(['x', 'y', 'z'])
    predicted.sort_values(by=['x','y','z'], inplace=True)

    size_before_drop_dup = len(predicted)
    predicted.drop_duplicates(inplace = True)
    size_after_drop_dup = len(predicted)

    if size_after_drop_dup < size_before_drop_dup:
        print(f"THERE WERE {size_before_drop_dup-size_after_drop_dup} DUPLICATES ON {file_path}")

    predicted.astype({'phi':np.float32})
    return predicted

def get_dataframe(file_path:pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = treat_predicted(df, file_path)

    return df

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

    if not first_file.exists() or not second_file.exists():
        print("First or second file doesnt exists!")
        exit(-1)
    
    first_predicted = get_dataframe(first_file)
    second_predicted = get_dataframe(second_file)

    first_len = len(first_predicted)
    print(f"first size: {len(first_predicted)}")
    second_len = len(second_predicted)
    print(f"second size: {len(second_predicted)}")

    if not second_len == first_len:
        print("WARNING! Prediction file sizes are different!")

        merge = None
        if second_len > first_len:
            merge = second_predicted.merge(first_predicted, on=['x','y','z'], how='left', indicator=True)
        else:
            merge = first_predicted.merge(second_predicted, on=['x','y','z'], how='left', indicator=True)
        
        DIFF_FILE = pathlib.Path("./difference.csv")
        EQUAL_FILE = pathlib.Path("./equal.csv")
        merge[merge['_merge'] == 'left_only'].to_csv(DIFF_FILE, index=None)
        merge[merge['_merge'] == 'both'].to_csv(EQUAL_FILE, index=None)
        print(f"Saved difference and equal files!")

    mergedDf = first_predicted.merge(second_predicted, on=['x','y','z'], how='outer', indicator=True)
    MERGED_FILE = pathlib.Path("./merged.csv")
    mergedDf.to_csv(MERGED_FILE, index=None)
    print(f"Merged df head:\n{mergedDf.head()}")
    print(f"Merged size: {len(mergedDf)}")

    if len(mergedDf) > second_len:
        print("WARNING! There are more points in a file than another! Look at mergedDf size!")
    
    total_pred_diff = sum(mergedDf['phi_x'] - mergedDf['phi_y'])
    if math.isclose(0.0, total_pred_diff):
        print("There is no prediction difference!")
    else:
        print("WARNKING! THERE IS A CONSIDERABLE PREDICTION DIFFERENCE!")
            
    print(f"TOTAL PRED DIFF: {total_pred_diff}") 