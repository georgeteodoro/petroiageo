import multiprocessing as mp
import time

import header
import expand
# import petro
# import apply3

# Constants
INIT_IT = 2

def main():
    iterations = 2
    window = 3

    for i in range(iterations):
        print("Preparing header")
        nwells = header.prepare_header(window)
        print("Expanding points")
        nwells = nwells + expand.data_aug(i + 1, window)
        print(len(nwells))
        # features_sets = petro(nwells)
        # f = prep(features_sets)...
        # v = apply3(nwells, f)
        # predition_values = prep2(v)

        # write(nwells)
        # write(predition_values)


if __name__ == '__main__':
    main()
