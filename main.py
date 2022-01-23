import multiprocessing as mp
import pandas as pd
import time

import header
import expand
import petro
# import apply3

# Constants
INIT_IT = 2

def main():
    iterations = 2
    window = 3
    
    for i in range(iterations):
        print("Preparing header")
        str_nwells = header.prepare_header(window)
        
        print("Expanding points")
        str_nwells += "\n" + expand.data_aug(i + 1, window) # param sA passed by global variable
        print(len(str_nwells))
        with open('tmp_data/nwells.csv', mode='w') as f:
            f.write(str_nwells)

        print("Performing feature selection")
        features_sets = petro.get_features_sets(str_nwells)
        print(features_sets)
        # f = prep(features_sets)...
        # v = apply3(nwells, f)
        # predition_values = prep2(v)
        # sA =+ values?

        # write(nwells)
        # write(predition_values)


if __name__ == '__main__':
    main()
