import numpy as np
import sys
from io import StringIO
import time

from functools import partial
import multiprocessing as mp

import global_variables

# Load global variables
snear = global_variables.snear
smid = global_variables.smid
sfar = global_variables.sfar
sufar = global_variables.sufar
sgersz = global_variables.sgersz
sgst = global_variables.sgst
sA = global_variables.porosity_values
seismic_shape = global_variables.seismic_shape
LABELS_SHAPE = global_variables.LABELS_SHAPE


# Arrays are 1D for being usable on shared-memory
# reshaping is too expensive, thus snear[i*shape[1]*shape[2] + j*shape[2] + k]
# Shape changing is too expensive: 20 sec for 100M and require 270M per array
def well_str(p, w, real, xx, shape):
    # locks = sLocks
    # visited = sVisited

    # Get well ID if it is real, otherwise 0
    # ID must begin at 1 (maybe not? probably not...)
    if p in real:
        r = real.index(p)
        # r = real.index(p) + 1
    else:
        r = 0

    s = StringIO()
    for k in range(251):  # for all depths
        # Calculate 1D coordinate of current point
        origPcoord = (p[0], p[1], k)
        pCoord = origPcoord[0] * LABELS_SHAPE[1] * LABELS_SHAPE[
            2] + origPcoord[1] * LABELS_SHAPE[2] + origPcoord[2]

        # # Checks for existing coordinate
        # locks[pCoord].acquire()
        # if visited[pCoord] != 0:
        #     # Marks first visit
        #     visited[pCoord] = 1
        #     locks[pCoord].release()
        # else:
        #     # Skips for already visited coordinates
        #     locks[pCoord].release()
        #     continue

        # Only expand points which have at least 0.05 porosity
        if sA[pCoord] <= 0.05:
            continue

        for i in range(p[0] - w, p[0] + w + 1):
            for j in range(p[1] - w, p[1] + w + 1):
                for z in range(k - w, k + w + 1):
                    # Deals with border cases
                    zz = min(max(z, 1), 250)

                    # Calculate 1D coordinate to be accessed by signal arrays
                    coord = i * shape[1] * shape[2] + j * shape[2] + zz

                    s.write("%s," % snear[coord])
                    s.write("%s," % smid[coord])
                    s.write("%s," % sfar[coord])
                    s.write("%s," % sufar[coord])
                    s.write("%s," % sgersz[coord])
                    s.write("%s," % sgst[coord])

        # imprime o id do poco, seja ele real ou aumentado, sao 10 pocos reais
        s.write("%s," % r)

        # define se o poco e real ou dado aumentado [...]
        if [p[0], p[1]] in real:
            s.write("1,")
        elif [p[0], p[1]] in xx:
            s.write("2,")
        else:
            s.write("0,")

        s.write("%s," % p[0])
        s.write("%s," % p[1])
        s.write("%s," % k)

        s.write("%s," % sA[pCoord])
        # Currently B, C and D labels were all zero
        s.write("0.0,")
        s.write("0.0,")
        s.write("0.0")
        s.write('\n')
    return s


def data_aug(iteration, window):
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

    # TODO: Add visited coords logic later (which suits parallel execution)
    # # coordenadas ja visitadas
    # coords = np.zeros((400, 700, 260)) # old
    # coords = np.zeros((434,646,251))   # with 2 more features
    # r = 1

    t2 = time.time()

    # Parallel execution
    f = partial(well_str, w=window, real=real, xx=xx, shape=seismic_shape)
    # with mp.Pool(1) as pool:
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(f, pp)

    t3 = time.time()

    # Compile results
    s_results = "".join([r.getvalue() for r in results])

    # with open('out.csv', mode='w') as f:
    #     for r in results:
    #         print(r.getvalue(), file=f)
    # for r in results:
    #     print(r.getvalue(), end='')

    t4 = time.time()

    with open('tt-times.log', mode='a') as f:
        print("iteration %s" % iteration, file=f)
        print("prep " + str(t2 - t1), file=f)
        print("exec " + str(t3 - t2), file=f)
        print("result " + str(t4 - t3), file=f)
        print("", file=f)

    # Return results as a string
    return s_results


if __name__ == '__main__':
    data_aug(sys.argv[1], 3)