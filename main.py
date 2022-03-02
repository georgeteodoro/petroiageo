import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from os import linesep
# from mpi4py import MPI
import seismic_data
import wells_data

import expand

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# mpi_size = comm.Get_size()
# manager_rank = mpi_size - 1

# Constants
INIT_IT = 2


def main():

    # Instantiate pandas dataframe for all data
    # Data structure is composed by:
    #   x,y,z(depth),
    #   well => Well ID (begins at 0? what if it is not real well point? )
    #                   (currently, -1 if it's not an original real point)
    #   real => [2=point to be expanded, 1=expanded point, 0=real well point]
    #   phi  => Porosity value
    #   rho  => ?
    #   vp   => ?
    #   vs   => ?

    # Read seismic data and add it to a dataframe
    features_columns = ["NEAR", "MID", "FAR", "UFAR", "GERSZ", "GST"]
    print("[main] Loading seismic data")
    features_df = seismic_data.get_all_seismic_data(features_columns)
    # print(features_df)

    # Real wells' data into a main dataframe
    print("[main] Loading wells values")
    main_df = wells_data.get_wells_data('./dados/porosity-canal.txt')

    # Add index of xyz and sort dataframe for better access times
    print("[main] Indexing all data by (x,y,z)")
    index = pd.MultiIndex.from_arrays(
        [main_df['x'], main_df['y'], main_df['z']])
    main_df.set_index(index, inplace=True)
    main_df.sort_index(inplace=True)

    # print("[main] Adding empty seismic features columns")
    # window = 3
    # new_cols = dict()
    # for col in features_columns:
    #     for i in range(-window, window + 1):
    #         for j in range(-window, window + 1):
    #             for k in range(-window, window + 1):
    #                 new_cols[f'{col}{i}{j}{k}'] = [np.nan]

    # print(f'new cols: {len(new_cols)}')
    # new_columns_df = pd.DataFrame(new_cols)
    # main_df = pd.merge(main_df, new_columns_df, how='cross')

    print(main_df)

    iterations = 3

    for it in range(iterations):
        print(f"Expanding points [{it}]")
        main_df = expand.data_aug(it + 1, main_df)
        main_df.sort_index(inplace=True)

        # UNWANTED_COLUMNS = ['real', 'well', 'x', 'y', 'z', 'phi']
        # all_features = main_df.columns.to_list()
        # for x in UNWANTED_COLUMNS:
        #     print(x)
        #     all_features.remove(x)
        # print(all_features)

        print(f"Performing feature selection [{it}]")
        features_sets = petro.get_features_sets(main_df, features_df)
        # features_sets = petro_dist.get_features_sets(str_nwells)
        print(features_sets)

        return

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
