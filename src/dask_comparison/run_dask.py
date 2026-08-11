from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import h5py

import numpy as np

from dask.distributed import Client, WorkerPlugin, as_completed

from feature_sel import test_new_feature


# ============================================================
# Configuration
# ============================================================

SCHEDULER_ADDRESS = "tcp://127.0.0.1:8786"
DATASET_PATH = Path("/home/will/git/petroiageo/data/POV/porosity_data.h5")
POROSITY_DSET_NAME = "p"
FEAT_DSET_NAME = "f"
DISP_WINDOW_X = 0
DISP_WINDOW_Y = DISP_WINDOW_X
DISP_WINDOW_Z = DISP_WINDOW_X
FEATURES_PATH = "/home/will/git/petroiageo/data/POV/h5_features"
FEATURES_LIST = ['FAR', 'FAR2']


# ============================================================
# 1. Configuration
# ============================================================

def config_gen() -> list[Configuration]:
    """
    Placeholder configuration generator.
    """

    configurations = []

    for feature in FEATURES_LIST:
        for i in range(-DISP_WINDOW_X, DISP_WINDOW_X + 1):
            for j in range(-DISP_WINDOW_Y, DISP_WINDOW_Y + 1):
                for k in range(-DISP_WINDOW_Z, DISP_WINDOW_Z + 1):
                    configurations.append((feature, i, j, k))

    return configurations


# ============================================================
# 2. Load dataset on each worker
# ============================================================

class DatasetPlugin(WorkerPlugin):
    """
    Loads the NumPy dataset once when a worker starts.

    Because we use one worker per node, this gives each node
    one copy/mapping of the dataset shared by its 20 threads.
    """

    def __init__(self, path: str):
        self.path = path

    def setup(self, worker):
        mpi_kwargs = {}
        worker.dataset_h5 = h5py.File(self.path,
                                   'r', **mpi_kwargs)
        worker.dataset_h5 = worker.dataset_h5[POROSITY_DSET_NAME]

        worker.cur_feature_id = 0


# ============================================================
# 3. Actual task
# ============================================================

def _merge_arrays(arr1, arr2, feature_id):
    f_name = f"f{feature_id}"
    lookup = {
        (row["x"], row["y"], row["z"]): row['f']
        for row in arr1
    }

    f_dtype = arr1.dtype["f"]
    result = np.empty(
        len(arr2),
        dtype=arr2.dtype.descr + [(f_name, f_dtype)]
    )

    for name in arr2.dtype.names:
        result[name] = arr2[name]

    result["f"] = [
        lookup.get(
            (row["x"], row["y"], row["z"]),
            0 # feature = 0 if value does not exist
        )
        for row in arr2
    ]


def single_feature_trial(config):
    from dask.distributed import get_worker
    feature_name, dx, dy, dz = config

    worker = get_worker()
    print(f"[single_feature_trial][{worker.id}] testing {config}")
    print(worker.dataset_h5)
    training_data = worker.dataset_h5[:]
    cur_feature_id = worker.cur_feature_id

    # load feature and apply displacement
    feature_path = f"{FEATURES_PATH}/{feature_name}.h5"
    print(f"[single_feature_trial][{worker.id}] reading feature {feature_path}")
    feature_data = h5py.File(feature_path, 'r')[FEAT_DSET_NAME]
    print(feature_data)
    feature_data['x'] += dx
    feature_data['y'] += dy
    feature_data['z'] += dz
    training_data = _merge_arrays(training_data, feature_data, cur_feature_id)

    # rmse, mae = test_new_feature(training_data)

    # print(test_new_feature())
    print(f"[single_feature_trial][{worker.id}] done {config}")

    return f"done from {worker}"

def commit_feature(feature_name, dx, dy, dz):
    worker = get_worker()
    training_data = worker.dataset_h5

    # load feature and apply displacement
    feature_data = np.load(feature_path)
    feature_data['x'] += dx
    feature_data['y'] += dy
    feature_data['z'] += dz
    training_data = _merge_arrays(training_data, feature_data, worker.cur_feature_id)

    worker.cur_feature_id += 1



# ============================================================
# 5. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Connect to scheduler
    # --------------------------------------------------------

    client = Client(SCHEDULER_ADDRESS)

    print(client)

    # --------------------------------------------------------
    # Load the dataset on every worker.
    #
    # IMPORTANT:
    #
    # This does NOT create a Dask task for the dataset.
    # It initializes worker-local state once.
    # --------------------------------------------------------

    client.register_plugin(
        DatasetPlugin(str(DATASET_PATH))
    )

    # --------------------------------------------------------
    # Generate configurations
    # --------------------------------------------------------

    configurations = config_gen()
    configurations = [configurations[0]]

    print(
        f"Generated {len(configurations)} configurations"
    )

    # --------------------------------------------------------
    # Discover workers.
    #
    # We expect exactly one worker per physical node.
    # --------------------------------------------------------

    scheduler_info = client.scheduler_info()

    workers = list(
        scheduler_info["workers"].keys()
    )

    print("\nWorkers:")

    for worker in workers:

        info = scheduler_info["workers"][worker]

        print(
            f"  {worker}: "
            f"{info['nthreads']} threads"
        )

    # --------------------------------------------------------
    # Submit exactly ONE task per configuration.
    #
    # run(config)
    #
    # Each task occupies one worker thread.
    # --------------------------------------------------------

    futures = []

    for task_params in configurations:
        future = client.submit(
            single_feature_trial,
            task_params,

            # Hard placement:
            # this task can execute ONLY on this worker.
            # workers=[worker],
            # allow_other_workers=False,

            # Optional explicit priority.
            priority=0,
        )

        futures.append(future)

    print(
        f"\nSubmitted {len(futures)} tasks"
    )

    # --------------------------------------------------------
    # Collect results as tasks finish.
    # --------------------------------------------------------

    results = []

    for future in as_completed(futures):

        result = future.result()

        results.append(result)

        print(
            f"Completed aff={result['aff']}"
        )

    print(
        f"\nCompleted {len(results)} tasks"
    )

    return results


if __name__ == "__main__":
    main()