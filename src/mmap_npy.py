import mmap
from math import prod
import psutil
# import os

import numpy as np

def mem_status(b_str):
    print(b_str)
    print(f'res: {psutil.Process().memory_info().rss/1024**3:.2f}, '
          f'shd: {psutil.Process().memory_info().rss/1024**3:.2f}')

def main():
    
# memmap([ 2.14554167,  1.88729522,  1.25540839,  0.36241464, -0.59173702,
#         -1.39025844, -1.85433537, -1.86001584, -1.4099719 ])


    mem_status('===== before open:')

    filename = '/home/will/git/petroiageo/data/POV/features/FAR.npy'
    f = open(filename, 'rb')
    mem_status('===== after open:')


    # file_size = os.path.getsize(filename)
    # print(file_size)

    np_version = np.lib.format.read_magic(f)
    np_header = np.lib.format._read_array_header(f, np_version)
    np_shape, _, np_type = np_header
    np_length = prod(np_shape) * np_type.itemsize
    # file_size -= 128

    # header = np.lib.format._read_bytes(f, header_length, "array header")
    import struct
    # header_size = 8*struct.calcsize(
    #     np.lib.format._header_size_info.get(np_version)[0])
    header_size = 128 # 16 bytes + padding for alignment

    # MAP_PRIVATE is ok since pages are copy-on-write and this file
    # is read-only
    
    src = mmap.mmap(f.fileno(), np_length+header_size, 
        flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE, 
        prot=mmap.PROT_READ)
    # src.seek(16)
    mem_status('===== after mmap:')

    print(src)
    print(np_shape)
    print(np_type)

    arr = np.ndarray(np_shape, np_type, buffer=src, offset=header_size)

    s = sum(sum(sum(arr == 0)))
    print(f'sum=0: {s}')
    mem_status('===== after check:')

    # del arr
    # mem_status('===== after del:')

    src.madvise(mmap.MADV_DONTNEED)
    mem_status('===== after madvise:')

    # src.close()
    # mem_status('===== after mmap close:')

    # # input('all done... (press enter)')
    # f.close()
    # mem_status('===== after file close:')



if __name__ == '__main__':
    main()


    #         feature_fd = open(feature_path, 0, "rb");
    # cache_el_t* src = mmap(NULL, filesize + 16,
    #     PROT_READ, MAP_PRIVATE | MAP_POPULATE, feature_fd, 0);
    #     memcpy(cache_line, src, cache_line_size);
    #     ret = posix_fadvise(feature_fd, 0, filesize, POSIX_FADV_DONTNEED);
    #     ret = munmap(src-16, filesize);
    #     ret = close(feature_fd);
