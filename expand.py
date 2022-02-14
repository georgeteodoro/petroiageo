import numpy as np
import sys
from io import StringIO
import time
import pandas as pd


# Using pandas DataFrame
def well_expand(p, real, xx, df):
    for z in range(251):  # for all depths
        x = p[0]
        y = p[1]

        # Only expand points which have at least 0.05 porosity
        if df.loc[(x, y, z), 'phi'] <= 0.05:
            continue

        # If this point was not real and is inside xx,
        # then it is an expanded point
        if df.loc[(x, y, z), 'real'] != 0 & ([x, y] in xx):
            df.loc[(x,y,z), 'real'] = 1



def data_aug(iteration, df):
    t1 = time.time()

    iteration = int(iteration)

    # Loads xx and pp non-initial values
    nxx = np.load('dados/xx.npy', allow_pickle=True)
    npp = np.load('dados/pp.npy', allow_pickle=True)

    # 0 is a placeholder for no-value on a sparse matrix
    def allButZero(arr):
        return arr[0:arr.index(0)]

    xx = allButZero(nxx[iteration].tolist())
    pp = allButZero(npp[iteration].tolist())

    # Clear numpy arrays
    nxx = None
    npp = None

    # esses sao pocos reais
    real = [[146, 500], [287, 242], [200, 102], [344, 276], [134, 227],
            [250, 315], [174, 365], [236, 113], [167, 186], [230, 194]]

    t2 = time.time()

    for point in pp:
        well_expand(point, real, xx, df)

    t3 = time.time()

    print(f'[expand] exp time: {t3-t2}')


if __name__ == '__main__':
    data_aug(sys.argv[1], 3)