"""
This script has the objective to filter a x-y area over
a larger area from a npy file. It is basically a slice.
"""
import numpy as np
import pathlib

if __name__ == "__main__":
    # No need to add a +1 on the ?_end to filter.
    # We already do that on the slice
    x_start, x_end = 0,0
    y_start, y_end = 0,0

    print(f"x range to filter: [{x_start}, {x_end}]")
    print(f"y range to filter: [{y_start}, {y_end}]")

    larger_area_path = pathlib.Path("")
    filtered_area_target_path = pathlib.Path("")

    print(f"Loading data from {larger_area_path}")
    larger_area = np.load(larger_area_path)
    print(f"Larger area shape: {larger_area.shape}")

    print(f"Filtering data")
    filtered_area = larger_area[x_start:x_end+1, y_start:y_end+1]
    print(f"Filtered area shape: {filtered_area.shape}")

    print(f"Saving filtered data at {filtered_area_target_path}")
    np.save(filtered_area_target_path, filtered_area)