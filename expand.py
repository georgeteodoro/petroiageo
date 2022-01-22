import numpy as np
import sys
from io import StringIO
import time
from math import prod

from functools import partial
import multiprocessing as mp

# solution based on:
# https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing

# LABELS_SHAPE = (400, 700, 260) # old
LABELS_SHAPE = (434, 646, 251)


# Arrays are 1D for being usable on shared-memory
# reshaping is too expensive, thus near[i*shape[1]*shape[2] + j*shape[2] + k]
# Shape changing is too expensive: 20 sec for 100M and require 270M per array
def well_str(p, w, real, xx, shape):
    # get shared memory variables
    near = snear
    mid = smid
    far = sfar
    ufar = sufar
    gersz = sgersz
    gst = sgst
    A = sA
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
        if A[pCoord] <= 0.05:
            continue

        for i in range(p[0] - w, p[0] + w + 1):
            for j in range(p[1] - w, p[1] + w + 1):
                for z in range(k - w, k + w + 1):
                    # Deals with border cases
                    zz = min(max(z, 1), 250)

                    # Calculate 1D coordinate to be accessed by signal arrays
                    coord = i * shape[1] * shape[2] + j * shape[2] + zz

                    s.write("%s," % near[coord])
                    s.write("%s," % mid[coord])
                    s.write("%s," % far[coord])
                    s.write("%s," % ufar[coord])
                    s.write("%s," % gersz[coord])
                    s.write("%s," % gst[coord])

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

        s.write("%s," % A[pCoord])
        # Currently B, C and D labels were all zero
        s.write("0.0,")
        s.write("0.0,")
        s.write("0.0")
        s.write('\n')
    return s


if __name__ == '__main__':

    # Labels b, c and d are useless (all zero)
    # Replaced all references to them (B[x]) to 0
    def fill_labels(iteration, f_name, a):
        with open(f_name, 'r') as f:
            for line in f.readlines():
                fields = line.split(' ')
                coord = int(
                    fields[0]) * LABELS_SHAPE[1] * LABELS_SHAPE[2] + int(
                        fields[1]) * LABELS_SHAPE[2] + int(fields[2])
                a[coord] = float(fields[3])

    t1 = time.time()

    iteration = int(sys.argv[1])

    r = 0
    # janela para variar as dimensoes do cubo
    w = 3

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

    def load_to_mem(filename):
        npa = np.load(filename)
        return npa.flatten(), npa.shape

    # Convert to list for better access time
    # numpy array is lazy with access
    near, nearshape = load_to_mem('dados/NEAR.npy')
    mid, midshape = load_to_mem('dados/MID.npy')
    far, farshape = load_to_mem('dados/FAR.npy')
    ufar, ufarshape = load_to_mem('dados/UFAR.npy')
    gersz, gerszshape = load_to_mem('dados/GERSZ.npy')
    gst, gstshape = load_to_mem('dados/GST.npy')

    # esses sao pocos reais
    real = [[146, 500], [287, 242], [200, 102], [344, 276], [134, 227],
            [250, 315], [174, 365], [236, 113], [167, 186], [230, 194]]

    # TODO: Add visited coords logic later (which suits parallel execution)
    # # coordenadas ja visitadas
    # coords = np.zeros((400, 700, 260)) # old
    # coords = np.zeros((434,646,251))   # with 2 more features
    # r = 1

    t2 = time.time()

    t21 = time.time()
    # print("shrd begin")

    labelslen = prod(LABELS_SHAPE)

    t22 = time.time()
    # print("labels done " + str(t22 - t21))

    # Allocate shared memory arrays
    # These are not thread-safe since read-only
    snear = mp.Array('d', len(near), lock=False)
    smid = mp.Array('d', len(mid), lock=False)
    sfar = mp.Array('d', len(far), lock=False)
    sufar = mp.Array('d', len(ufar), lock=False)
    sgersz = mp.Array('d', len(gersz), lock=False)
    sgst = mp.Array('d', len(gst), lock=False)
    sA = mp.Array('d', labelslen, lock=False)
    # sVisited = mp.Array('d', labelslen, lock=False)
    # sLocks = mp.RawArray(type(mp.Lock()), labelslen)

    t23 = time.time()

    # print("arrays done " + str(t23 - t22))

    # Fill values on shared memory space
    # Best if input arrays come from lazy numpy.array
    # since this assignment is a deep-copy to shared
    # memory space.
    def move_to_shrd(arrs):
        # Shared arrays can only be accessed like this
        sarrs = [snear, smid, sfar, sufar, sgersz, sgst]
        arr, _ = load_to_mem(arrs[1])
        sarrs[arrs[0]][:] = arr

    # The move to shared memory is done in parallel.
    # Magic numbers are IDs for a list with the shared arrays
    # which is inside move_to_shrd.
    # Name of files passed so they can be opened inside move_to_shrd, which
    # is the only way to "pass" the opened np.arrays to a parallel worker.
    with mp.Pool(6) as pool:
        pool.map(move_to_shrd, [[0, "dados/NEAR.npy"], [1, "dados/MID.npy"],
                                [2, "dados/FAR.npy"], [3, "dados/UFAR.npy"],
                                [4, "dados/GERSZ.npy"], [5, "dados/GST.npy"]])
    t24 = time.time()
    # print("cpy done " + str(t24 - t23))

    fill_labels(iteration, sys.argv[2], sA)

    # Create a shared structure for synchronizing sVisited array by coordinate
    # sVisited[:] = np.zeros((labelslen))

    t25 = time.time()
    # print("shrd end " + str(t25 - t24))

    t3 = time.time()

    # Parallel execution
    f = partial(well_str, w=w, real=real, xx=xx, shape=nearshape)
    # with mp.Pool(1) as pool:
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(f, pp)

    t4 = time.time()

    # Write results
    # with open('out.csv', mode='w') as f:
    #     for r in results:
    #         print(r.getvalue(), file=f)
    for r in results:
        print(r.getvalue(), end='')

    t5 = time.time()

    with open('tt-times.log', mode='a') as f:
        print("iteration %s" % iteration, file=f)
        print("prep " + str(t2 - t1), file=f)
        print("shm-prep " + str(t3 - t2), file=f)
        print("exec " + str(t4 - t3), file=f)
        print("write " + str(t5 - t4), file=f)
        print("", file=f)
