import time
import numpy as np
from functools import reduce
import sys
import random

# max_features = 1372 # for w1
# max_features = 500 # for w2

# 500 value is legacy:
# Old input file 'ids' had values between 1-500
# Somewhere in exec.sh 7x7x7 features area becomes 5x5x5
# Should we do something about the first iteration, which is 7x7x7?
MAX_ID = 500

# Only this number of features is chosen
N_FEATURES = 15

# Standard data fields, e.g., phi, rho, ...
N_INFO = 9

# Random numbers seed
RANDOM_STATE = 1

t1 = time.time()

# Get inputs
max_features = int(sys.argv[1])
wells_file = sys.argv[2]
output_file = sys.argv[3]

# Generate random features' ids
random.seed(RANDOM_STATE)
ids = random.sample(range(1,MAX_ID+1), N_FEATURES)

# Allocate npy data
num_lines = sum(1 for line in open(wells_file, 'r'))-1  # last line is empty
output = np.empty((num_lines, N_FEATURES + N_INFO))

t2 = time.time()

# Filter full data to reduced features
with open(wells_file, 'r') as f:
    # First line is labels only
    all_labels = f.readline()
    
    for i in range(num_lines):
        # Move selected features features
        fields = f.readline().split(',')
        for j in range(N_FEATURES):
            output[i][j] = fields[ids[j] - 1]

        # Move info data fields
        output[i][N_FEATURES:] = fields[max_features:max_features+N_INFO]

t3 = time.time()

# Write results
with open(output_file, 'w') as f:
    all_labels = all_labels.replace('\n', '').split(',')
    labels = reduce(lambda rem, id: rem + ',' + all_labels[id - 1], ids,
                    "")[1:]
    labels += reduce(lambda rem, id: rem + ',' + all_labels[id - 1],
                     range(max_features+1, max_features + N_INFO + 1), "")
    print(labels, file=f)
    for i in range(num_lines):
        print(reduce(lambda x, y: str(x) + ',' + str(y), output[i]), file=f)

t4 = time.time()

# Write profiling times
with open('prep1-times.log', mode='a') as f:
        print("prep: " + str(t2 - t1), file=f)
        print("process: " + str(t3 - t2), file=f)
        print("write: " + str(t4 - t3), file=f)
        print("", file=f)
