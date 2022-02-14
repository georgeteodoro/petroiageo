import multiprocessing as mp
import pandas as pd
import time
from collections import defaultdict
from os import linesep
from mpi4py import MPI
import seismic_data
import wells_data

import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
manager_rank = mpi_size - 1

# Only import for manager
if rank == manager_rank:
    # import global_variables
    # import header
    # import expand
    # import petro
    # import petro_dist
    # import apply3

    # Constants
    INIT_IT = 2
    # LABELS_SHAPE = global_variables.LABELS_SHAPE


def main():

    # Instantiate pandas dataframe for all data
    # Data structure is composed by:
    #   x,y,depth,
    #   seismic features,
    #   well => Well ID (begins at 0? what if it is not real well point? )
    #                   (currently, -1 if it's not an original real point)
    #   real => [2=point to be expanded, 1=expanded point, 0=real well point]
    #   phi  => Porosity value
    #   rho  => ?
    #   vp   => ?
    #   vs   => ?

    # Read seismic data and add it to a dataframe
    seismic_columns = ["NEAR", "MID", "FAR", "UFAR", "GERSZ", "GST"]
    print("[main] Loading seismic data")
    seismic_df = seismic_data.get_all_seismic_data(seismic_columns)
    # print(seismic_df.head(4))

    print("[main] Loading wells values")
    df = wells_data.merge_wells_data(seismic_df, './dados/porosity-canal.txt')

    print("[main] Indexing all data by (x,y,z)")
    index = pd.MultiIndex.from_arrays([df['x'], df['y'], df['z']])
    df.set_index(index, inplace=True)

    print(df)

    return

    iterations = 3
    window = 3

    for it in range(iterations):
        print(f"Preparing header [{it}]")
        str_nwells = header.prepare_header(window)

        print(f"Expanding points [{it}]")
        str_nwells += "\n" + expand.data_aug(
            it + 1, window)  # param sA passed by global variable
        with open(f"tmp_data/nwells-{it}.csv", mode='w') as f:
            f.write(str_nwells)

        print(f"Performing feature selection [{it}]")
        features_sets = petro.get_features_sets(str_nwells)
        # features_sets = petro_dist.get_features_sets(str_nwells)
        # print(features_sets)

        print(f"Performing predictions [{it}]")
        # Sort by second column (id 1)
        features_sets.sort(key=lambda tup: tup[1], reverse=True)
        best_features_set = features_sets[0][0]
        v = apply3.perf_predition(best_features_set, str_nwells)
        with open(f'tmp_data/v-{it}', mode='w') as f:
            f.write("".join([
                f"{p}\n".replace('[', '').replace(']', '').replace(',', '')
                for p in v
            ]))

        print(f"Expanding predictions [{it}]")
        v2 = defaultdict(list)
        # Create a dict of all predictions per point
        for p in v:
            for i in range(p[0] - 1, p[0] + 2):
                for j in range(p[1] - 1, p[1] + 2):
                    v2[f"{i} {j} {p[2]}"].append(p[3])

        # Averages the predictions of each point
        values = []
        for k, l in v2.items():
            values.append(k.split(" ") + [sum(l) / len(l)])
        with open(f'tmp_data/values{it}', mode='w') as f:
            f.write("".join([f"{p}\n" for p in values]))

        print(f"Updating porosity points [{it}]")
        for p in values:
            coord = int(p[0]) * LABELS_SHAPE[1] * LABELS_SHAPE[2] + int(
                p[1]) * LABELS_SHAPE[2] + int(p[2])
            global_variables.porosity_values[coord] = float(p[3])


if __name__ == '__main__':
    main()
