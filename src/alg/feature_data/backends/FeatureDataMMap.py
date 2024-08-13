import mmap
import numpy as np
from abc import ABC, abstractmethod
from math import prod
from time import time

from feature_data.backends.FeatureDataBase import FeatureDataBase


class FeatureDataMMap(FeatureDataBase):
    '''Implementation for in-memory feature through mmap.

    File-backed data should be .npy since this is a binary format.

    Page cache management is done at kernel level, however we help it
    through advises. Multiple intra-node processes can access the same
    pages. After the last process, the cache pages are advised to be 
    reclaimed by the kernel.

    On feature loading, a write lock is acquired, released after the 
    feature is loaded. Before the np.ndarray creation, a read lock is 
    acquired. The read lock is released at destruction. All locking is
    blocking, thus, to avoid waiting a possible feature loading process 
    futures are recommended.
    '''

    def __init__(self,
                 feature_path,
                 done_reading_callback=None,
                 pre_fetch=False):
        super(FeatureDataMMap, self).__init__()

        # Callback of cache manager to be called at deletion
        # This signals that this current object is no longer
        # reading the input feature.
        self._done_reading_callback = done_reading_callback

        # Open file
        ext = feature_path.split('/')[-1].split('.')[-1]
        assert ext == 'npy', f"[FeatureDataMMap] "\
                              f"File is not numpy: {feature_path}"
        self._file = open(feature_path, 'rb')

        # Retrieve numpy header data
        np_version = np.lib.format.read_magic(self._file)
        np_header = np.lib.format._read_array_header(self._file, np_version)
        np_shape, _, np_type = np_header
        np_length = prod(np_shape) * np_type.itemsize
        # TODO: get this from .npy file
        np_header_size = 128  # 16 bytes + padding for alignment

        # Pre-load the whole data into cache pages
        flags = mmap.MAP_PRIVATE | mmap.MADV_SEQUENTIAL
        if pre_fetch:
            flags |= mmap.MAP_POPULATE
        print("[FeatureDataMMap][__init__] mmapping...")
        t0 = time()
        self._mmap_buffer = mmap.mmap(self._file.fileno(),
                                      np_length + np_header_size,
                                      flags=flags,
                                      prot=mmap.PROT_READ)
        t1 = time()
        print(f"[FeatureDataMMap][__init__] mmap_done {t1-t0:.4f}")

        # Create a npy array to wrap this memory buffer
        self._feature = np.ndarray(np_shape,
                                   np_type,
                                   buffer=self._mmap_buffer,
                                   offset=np_header_size)
        print(f"[FeatureDataMMap][__init__] ndarray_done {t2-t1:.4f}")
        t2 = time()

    def __del__(self):
        # Bug fix for interaction with mpi and page caching:
        # Sometimes unused cache pages are not reclaimed by kernel,
        # resulting in bus errors when reading a new feature.
        # Ideally, self._mmap_buffer.close() should be enough.
        self._mmap_buffer.madvise(mmap.MADV_DONTNEED)

        self._mmap_buffer.close()
        self._file.close()

        if self._done_reading_callback is not None:
            self._done_reading_callback()

    def filter_coords(self, coords):
        '''
        Filter points, one by one. No performance guarantee was made with this
        simple implementation.
        Do we need a generator???
        '''

        # Allocate output array
        # print(f'[FeatureDataMMap][filter_coords] alloc len {len(coords)}')
        points = np.empty((len(coords), ), np.float64)

        # print(f'[FeatureDataMMap][filter_coords] filtering')
        for (i, c) in enumerate(coords):
            points[i] = self._feature[tuple(c)]

        # print(f'[FeatureDataMMap][filter_coords] filtering done')

        return points