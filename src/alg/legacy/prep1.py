import time
from functools import reduce
import sys
import random
from io import StringIO
from functools import partial
import multiprocessing as mp

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
ids = random.sample(range(MAX_ID), N_FEATURES)

# Get number of lines
num_lines = sum(1 for line in open(wells_file, "r")) - 1  # last line is empty
# output = []

t2 = time.time()


def filter_line(line, max_features, ids):
    s = StringIO()

    # Move selected features features
    fields = line.split(",")
    for i in range(N_FEATURES):
        s.write(str(fields[ids[i]]) + ",")

    # Move info data fields
    for f in fields[max_features : max_features + N_INFO]:
        s.write(str(f) + ",")

    # Return output without trailing comma
    return s.getvalue()[:-1]


# Filter full data to reduced features
with open(wells_file, "r") as f:
    # First line is labels only
    all_labels = f.readline()

    # Parallel execution for each line of f
    with mp.Pool(mp.cpu_count()) as pool:
        output = pool.map(
            partial(filter_line, max_features=max_features, ids=ids),
            f.readlines(),
        )

    # output = []
    # for line in f.readlines():
    #     output.append(filter_line(line, max_features, ids))

t3 = time.time()

# Write results
with open(output_file, "w") as f:
    all_labels = all_labels.replace("\n", "").split(",")
    labels = reduce(lambda rem, id: rem + "," + all_labels[id], ids, "")[1:]
    labels += reduce(
        lambda rem, id: rem + "," + all_labels[id],
        range(max_features, max_features + N_INFO),
        "",
    )
    print(labels, file=f)

    for o in output:
        print(o, file=f)

t4 = time.time()

# Write profiling times
# with open('prep1-times.log', mode='a') as f:
print("prep: " + str(t2 - t1))
print("process: " + str(t3 - t2))
print("write: " + str(t4 - t3))
# print("", file=f)
