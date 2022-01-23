import multiprocessing as mp
import time

import header
import expand
# import petro
# import apply3

# Constants
INIT_IT = 2
# LABELS_SHAPE = (434, 646, 251) # also on

# def read_porosities(filename, a):
#     with open(filename, 'r') as f:
#         for line in f.readlines():
#             fields = line.split(' ')
#             coord = int(fields[0]) * LABELS_SHAPE[1] * LABELS_SHAPE[2] + int(
#                 fields[1]) * LABELS_SHAPE[2] + int(fields[2])
#             a[coord] = float(fields[3])


def main():
    iterations = 1
    window = 3

    # Prepare porosity values array and read initial data
    # porosity_values = mp.Array('d', expand.get_labels_len(), lock=False)
    # read_porosities("dados/porosity-canal.txt", porosity_values)

    for i in range(iterations):
        print("Preparing header")
        nwells = header.prepare_header(window)
        print("Expanding points")
        # nwells = nwells + expand.data_aug(i + 1, window, porosity_values)
        nwells = nwells + expand.data_aug(i + 1, window,
                                          f"tmp_data/nwells-{i}.csv")
        print(nwells)
        # features_sets = petro(nwells)
        # f = prep(features_sets)...
        # v = apply3(nwells, f)
        # predition_values = prep2(v)

        # write(nwells)
        # write(predition_values)


if __name__ == '__main__':
    main()
