import numpy as np
import sys
from io import StringIO
import time
import pandas as pd
import warnings


# Using pandas DataFrame
def well_expand(p, real, xx, main_df):
    # List of points with phi=0 to be appended later
    # Inserting them later as a single DataFrane is quicker than inserting
    # one at a time.
    # Empirically, zero_coords list or array are similar in performance
    zero_coords = []

    # Disable PerformanceWarning for not sorting main_df
    # In practice, for this algorithm, not sorting is faster
    warnings.simplefilter(
        action="ignore", category=pd.errors.PerformanceWarning
    )

    for z in range(251):  # for all depths
        x = p[0]
        y = p[1]

        # If point is not present on main_df create it
        if len(main_df.index.intersection(pd.Index([(x, y, z)]))) == 0:
            zero_coords.append([x, y, z, -1, 1, 0])
            continue

        # Only expand existing points which have at least 0.05 porosity
        if main_df.loc[(x, y, z), "phi"] <= 0.05:
            continue

        # If this point was not real and is inside xx,
        # then it is an expanded point
        if main_df.loc[(x, y, z), "real"] != 0 & ([x, y] in xx):
            main_df.loc[(x, y, z), "real"] = 1

    zero_coords_df = pd.DataFrame(
        zero_coords, columns=["x", "y", "z", "well", "real", "phi"]
    )
    index = pd.MultiIndex.from_arrays(
        [zero_coords_df["x"], zero_coords_df["y"], zero_coords_df["z"]]
    )
    zero_coords_df.set_index(index, inplace=True)
    main_df = pd.concat([main_df, zero_coords_df])

    # Re-enabling warnings
    warnings.simplefilter(
        action="default", category=pd.errors.PerformanceWarning
    )

    return main_df


def data_aug(iteration, main_df):
    t1 = time.time()

    iteration = int(iteration)

    # TODO: xx and pp array is incremental, i.e., xx[3] in xx[4]
    # This means that expanded points are revisited for every iteration.
    # Updating xx.npy and pp.npy for intersection(xx[3], xx[4])=[] should
    # improve performance

    # Loads xx and pp non-initial values
    nxx = np.load("dados/xx.npy", allow_pickle=True)
    npp = np.load("dados/pp.npy", allow_pickle=True)

    # 0 is a placeholder for no-value on a sparse matrix
    def allButZero(arr):
        return arr[0 : arr.index(0)]

    xx = allButZero(nxx[iteration].tolist())
    pp = allButZero(npp[iteration].tolist())

    # Clear numpy arrays
    nxx = None
    npp = None

    # esses sao pocos reais
    real = [
        [146, 500],
        [287, 242],
        [200, 102],
        [344, 276],
        [134, 227],
        [250, 315],
        [174, 365],
        [236, 113],
        [167, 186],
        [230, 194],
    ]

    t2 = time.time()

    for point in pp:
        main_df = well_expand(point, real, xx, main_df)

    t3 = time.time()

    print(f"[expand] exp time: {t3-t2}")

    return main_df


if __name__ == "__main__":
    data_aug(sys.argv[1], 3)
