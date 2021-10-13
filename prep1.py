import time
import numpy as np
from functools import reduce
import sys

N_FEATURES = 15
# MAX_FEATURES = 1372 # for w1
# # MAX_FEATURES = 500 # for w2
N_INFO = 9

# get inputs
MAX_FEATURES = int(sys.argv[1])
IDS_FILE = sys.argv[2]
WELLS_FILE = sys.argv[3]
OUTPUT_FILE = sys.argv[4]

# values from [1,500]
# change it to a simple random number generator (ordered)
ids = []
with open(IDS_FILE, 'r') as f:
    for l in f.readlines():
        ids.append(int(l))
print(len(ids))
# ids = ids[:N_FEATURES]

# read wells data
num_lines = sum(1 for line in open(WELLS_FILE, 'r'))-1  # last line is empty
print("num_lines: " + str(num_lines))
output = np.empty((num_lines, N_FEATURES + N_INFO))
with open(WELLS_FILE, 'r') as f:
    all_labels = f.readline()
    for i in range(num_lines):
        # Get features
        fields = f.readline().split(',')
        for j in range(N_FEATURES):
            output[i][j] = fields[ids[j] - 1]

        # Get info data
        output[i][N_FEATURES:] = fields[MAX_FEATURES:MAX_FEATURES+N_INFO]

with open(OUTPUT_FILE, 'w') as f:
    all_labels = all_labels.replace('\n', '').split(',')
    labels = reduce(lambda rem, id: rem + ',' + all_labels[id - 1], ids,
                    "")[1:]
    labels += reduce(lambda rem, id: rem + ',' + all_labels[id - 1],
                     range(MAX_FEATURES+1, MAX_FEATURES + N_INFO + 1), "")
    print(labels, file=f)
    for i in range(num_lines):
        print(reduce(lambda x, y: str(x) + ',' + str(y), output[i]), file=f)

