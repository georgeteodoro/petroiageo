import random
from FeatureSchedFLoc import FeatureSchedFLoc
from FeatureSchedFIFO import FeatureSchedFIFO
from collections import defaultdict

mean = 10
stdev = 2
random.seed(92)

def sample_rutime():
    return random.gauss(mean, stdev)

worker_remaining_time = {}
worker_sched_feature = {}
node_cache = defaultdict(lambda: [])
max_cache_size = 2
N_FEATURES = 80
WINDOW = 1
N_DISPS = (WINDOW*2+1)**3
N_TOTAL_FEATURES = N_DISPS * N_FEATURES
WORKERS_PER_NODE = 48
N_NODES = 16
N_WORKERS = N_NODES * WORKERS_PER_NODE 

waiting_workers = list(range(N_WORKERS))
mpi_rank_mapping = {}
for n in range(N_NODES):
    local_workers = list(range(n*WORKERS_PER_NODE, (n+1)*WORKERS_PER_NODE))
    mpi_rank_mapping[n] = local_workers

node_of_rank = {}
for n, ranks in mpi_rank_mapping.items():
    for r in ranks:
        node_of_rank[r] = n

# workers type: (node, worker_id_per_node)
# feature type: (feature, disp)

class Config:
    def __init__(self):
        self.configs = {}
        self.alg = {'window': WINDOW}
        self.features_files_names = [str(f"F{i}") for i in range (N_FEATURES)]
    def set_param(self, key, value):
        self.configs[key] = value
    def get_param(self, key):
        return self.configs[key]

config = Config()
config.set_param('mpi_rank_mapping', mpi_rank_mapping)
config.set_param('max_feats_for_trial', -1)
config.set_param('fsched_debug', True)
config.set_param('num_features', -1)
config.set_param('small_window', False)

scheduler = FeatureSchedFLoc(config)
# scheduler = FeatureSchedFIFO(config)
scheduler.begin_iteration()

cache_hits = 0
cache_misses = 0
done_features = 0

while done_features < N_TOTAL_FEATURES:
    # alloc
    while len(waiting_workers) > 0:
        worker = waiting_workers.pop()
        node = node_of_rank[worker]
        task = scheduler.get_feature(worker)
        if task is None:
            # no more tasks
            continue
        feature, disp = task

        worker_remaining_time[worker] = sample_rutime()
        worker_sched_feature[worker] = task

        print(f"Sched {task} -> {node}:{worker}")

        # check cache hit
        if feature in node_cache[node]:
            print("\thit")
            cache_hits += 1
            # Place feature as recently used
            node_cache[node].remove(feature)
            node_cache[node].append(feature)        
        else:
            print("\tmiss")
            cache_misses += 1
            node_cache[node].append(feature)
            if len(node_cache[node]) > max_cache_size:
                # remove LRU
                node_cache[node].pop(0)

    # pass the time
    min_time = min(worker_remaining_time.values())
    to_remove = []
    for worker in range(N_WORKERS):
        if worker in worker_remaining_time:
            worker_remaining_time[worker] -= min_time
            assert worker_remaining_time[worker] >= 0
            # task is over
            if worker_remaining_time[worker] == 0:
                to_remove.append(worker)

    # finish tasks with 0 time
    for worker in to_remove:
        feature = worker_sched_feature[worker]
        worker_remaining_time.pop(worker)
        print(f"Worker {node}:{worker} done with {feature}")
        scheduler.tried_feature(feature, worker)
        waiting_workers.append(worker)
        done_features += 1

print(f"hits: {cache_hits}")
print(f"misses: {cache_misses}")

