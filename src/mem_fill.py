import numpy as np
import sys
import time

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} MEM_GB TIME_HRS")
        return

    s_x = 1000
    s_y = 1000
    s_z = 100
    shape = (s_x, s_y, s_z)

    expected_fill = float(sys.argv[1])
    size_per_fill = (8 * s_x * s_y * s_z) / 1024**3

    n_fills = int(expected_fill // size_per_fill)

    print(f'[mem_fill] Filling {n_fills*size_per_fill:.2f}GB with {n_fills} fills')

    all_fill = []
    size = 0
    for _ in range(n_fills):
        fill = np.full(shape, 5)
        all_fill.append(fill)
        # size += fill.itemsize * fill.size / 1024**3

    # a = input(f'waiting with size {size:.2f}GB...')

    #hundr_hrs = 100 * 3600
    print('[mem_fill] Sleeping...')
    time.sleep(3600 * sys.argv[2])
    print('[mem_fill] Awaken...')


if __name__ == '__main__':
    main()
