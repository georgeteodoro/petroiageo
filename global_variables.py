import numpy as np
import multiprocessing as mp
from math import prod

# solution based on:
# https://stackoverflow.com/questions/1675766/combine-pool-map-with-shared-memory-array-in-python-multiprocessing

# LABELS_SHAPE = (400, 700, 260) # old
LABELS_SHAPE = (434, 646, 251)


def load_to_mem(filename):
    npa = np.load(filename)
    return npa.flatten(), npa.shape


def get_labels_len():
    return prod(LABELS_SHAPE)


def read_porosities(filename, a):
    with open(filename, 'r') as f:
        for line in f.readlines():
            fields = line.split(' ')
            coord = int(fields[0]) * LABELS_SHAPE[1] * LABELS_SHAPE[2] + int(
                fields[1]) * LABELS_SHAPE[2] + int(fields[2])
            a[coord] = float(fields[3])


# Prepare porosity values array and read initial data
print("[global_variables] Reading initial porosity values")
porosity_values = mp.Array('d', get_labels_len(), lock=False)
read_porosities("dados/porosity-canal.txt", porosity_values)

# Convert to list for better access time
# numpy array is lazy with access
print("[global_variables] Preparing seismic data")
near, nearshape = load_to_mem('dados/NEAR.npy')
mid, midshape = load_to_mem('dados/MID.npy')
far, farshape = load_to_mem('dados/FAR.npy')
ufar, ufarshape = load_to_mem('dados/UFAR.npy')
gersz, gerszshape = load_to_mem('dados/GERSZ.npy')
gst, gstshape = load_to_mem('dados/GST.npy')
seismic_shape = nearshape

# Prepare seismic data arrays
global snear
global smid
global sfar
global sufar
global sgersz
global sgst
snear = mp.Array('d', len(near), lock=False)
smid = mp.Array('d', len(mid), lock=False)
sfar = mp.Array('d', len(far), lock=False)
sufar = mp.Array('d', len(ufar), lock=False)
sgersz = mp.Array('d', len(gersz), lock=False)
sgst = mp.Array('d', len(gst), lock=False)


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
print("[global_variables] Reading seismic data")
# with mp.Pool(6) as pool:
#     print("start")
#     pool.map(
#         move_to_shrd,
#         [[0, "dados/NEAR.npy"], [1, "dados/MID.npy"], [2, "dados/FAR.npy"],
#          [3, "dados/UFAR.npy"], [4, "dados/GERSZ.npy"], [5, "dados/GST.npy"]])
#     print("done")

# python's multiprocessing requires a __main__ function since it imports
# this module. Otherwise, it lingers in infinite recursion of loading this
# module, executing the map function and importing this module again
# Current solution is serialized, but it is ok since it only runs once
for d in [[0, "dados/NEAR.npy"], [1, "dados/MID.npy"], [2, "dados/FAR.npy"],
          [3, "dados/UFAR.npy"], [4, "dados/GERSZ.npy"], [5, "dados/GST.npy"]]:
    move_to_shrd(d)

# Clear numpy arrays
near = None
mid = None
far = None
ufar = None
gersz = None
gst = None