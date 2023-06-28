import h5py
import numpy as np
from math import prod
import gc
import psutil, os


def small_test():
    with h5py.File("chunked3d.h5", "w") as h5_f:
        # x, y, z
        shape = (4, 6, 3)
        chunk_shape = (2, 3, 3)
        vals = np.zeros(shape, dtype=np.int64)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    vals[i, j, k] = k + shape[0] * j + shape[0] * shape[1] * i

        vals = [
            k + shape[0] * j + shape[0] * shape[1] * i
            for i in range(shape[0])
            for j in range(shape[1])
            for k in range(shape[2])
        ]
        h5_dset = h5_f.create_dataset(
            "chunked", shape, dtype=np.int64, chunks=chunk_shape, data=vals
        )

        print(h5_dset.chunks)
        print(h5_dset[0, 0, :])


def large_io_test():
    with h5py.File("chunked3d.h5", "w") as h5_f:
        # Read feature file
        far_np = np.load("../dados/FAR.npy")
        shape = far_np.shape
        chunk_shape = (int(shape[0] / 4), int(shape[1] / 4), shape[2])

        print(
            f"mem1: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}"
        )

        h5_dset = h5_f.create_dataset(
            "large", shape, dtype=np.float64, chunks=chunk_shape, data=far_np
        )
        del far_np
        del h5_dset

    print(f"collected: {gc.collect()}")

    with h5py.File("chunked3d.h5", "r") as h5_f:
        h5_dset = h5_f["large"]
        print(h5_dset.chunks)
        col1 = h5_dset[0 : chunk_shape[0], 0 : chunk_shape[1], :]
        print(f"read: {col1.shape}")
        print(
            f"mem2: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}"
        )


if __name__ == "__main__":
    large_io_test()
