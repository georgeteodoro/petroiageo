from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import h5py
from tqdm import tqdm
import numpy as np

from dask.distributed import Client, WorkerPlugin, as_completed
import dask
import dask.array as da

from feature_sel import test_new_feature


# ============================================================
# Configuration
# ============================================================

BASE_PATH="/home/will/git/petroiageo"
# BASE_PATH="/snfs2/willianjunior/git/petroiageo"

SCHEDULER_ADDRESS = "tcp://127.0.0.1:8786"
# DATASET_PATH = Path(BASE_PATH + "/data/POV/porosity_data.h5")
DATASET_PATH = BASE_PATH + "/data/POV/raw/porosity-canal.txt"
POROSITY_DSET_NAME = "p"
FEAT_DSET_NAME = "f"
DISP_WINDOW_X = 0
DISP_WINDOW_Y = DISP_WINDOW_X
DISP_WINDOW_Z = DISP_WINDOW_X
FEATURES_PATH = BASE_PATH + "/data/POV/features/h5_features"
FEATURES_LIST = ['FAR', 'FAR2']

NUM_SEL_FEATURES = 2

REAL_WELLS =[(134, 227),(146, 500),(167, 186),(174, 365),(200, 102),(236, 113),(250, 315),(287, 242),(230, 194),(344, 276)]

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)

RealValues = enum("real", "propagated", "canal", "canal_expanded", "expanded",
                  "empty", "none")

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


def single_feature_trial(training_data, features_data, feature_name, f_iteration):
    from dask.distributed import get_worker

    feature = features_data[feature_name]
    training_data[f'f_{f_iteration}'] = feature

    worker = get_worker()
    # print(f"[single_feature_trial][{worker.id}] testing {config}")
    # print(worker.dataset_h5)
    # training_data = worker.dataset_h5
    # training_data = training_data[training_data['real'] != RealValues.empty]
    # print(f"[single_feature_trial][{worker.id}] done copying")
    # cur_feature_id = worker.cur_feature_id

    # # load feature and apply displacement
    # feature_path = f"{FEATURES_PATH}/{feature_name}.h5"
    # print(f"[single_feature_trial][{worker.id}] reading feature {feature_path}")
    # feature_data = h5py.File(feature_path, 'r')[FEAT_DSET_NAME]
    # print(feature_data)
    # feature_data['x'] += dx
    # feature_data['y'] += dy
    # feature_data['z'] += dz
    # training_data = _merge_arrays(training_data, feature_data, cur_feature_id)

    rmse, mae = test_new_feature(training_data, f_iteration)

    # print(test_new_feature())
    print(f"[single_feature_trial][{worker.id}][f_it{f_iteration}] done {feature_name}: {rmse}, {mae}")

    return feature_name, rmse, mae

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

    sel_f_types = []
    for i in range(NUM_SEL_FEATURES):
        sel_f_types.append((f'f_{i}', np.float64))

    dtype = np.dtype([
        ("x", np.int32),
        ("y", np.int32),
        ("z", np.int32),
        ("phi", np.float32),
        ("real", np.int32),
        ("well_id", np.int32),
    ] + sel_f_types)
    raw = np.loadtxt(DATASET_PATH)
    data = np.empty(
        len(raw),
        dtype=dtype,
    )
    data["x"] = raw[:, 0]
    data["y"] = raw[:, 1]
    data["z"] = raw[:, 2]
    data["phi"] = raw[:, 3]
    data["well_id"] = -1
    data["real"] = RealValues.real

    real_wells_dict = {}
    for i, w in enumerate(REAL_WELLS):
        real_wells_dict[w] = i

    print(real_wells_dict)

    for i, d in enumerate(data):
        x, y = int(d['x']), int(d['y'])
        if (x, y) in real_wells_dict:
            data[i] = real_wells_dict[(x,y)]

    training_data = data
    # training_data_da = da.from_array(training_data)
    # print(training_data_da)

    # get coordinates of current points
    coordinates = training_data[['x','y','z']]
    print(coordinates)

    # Prepare the features
    all_feature_types = []
    mpi_kwargs = {}
    features = []
    feature_files = {}
    for feature_name in FEATURES_LIST:
        feature_file = h5py.File(FEATURES_PATH + "/" + feature_name + ".h5",
                                'r', **mpi_kwargs)
        feature_files[feature_name] = feature_file[FEAT_DSET_NAME]
        for dx in range(-DISP_WINDOW_X, DISP_WINDOW_X + 1):
            for dy in range(-DISP_WINDOW_Y, DISP_WINDOW_Y + 1):
                for dz in range(-DISP_WINDOW_Z, DISP_WINDOW_Z + 1):
                    features.append((feature_name, dx, dy, dz))
                    all_feature_types.append((f'{feature_name}-{dx}-{dy}-{dz}', np.float64))

    dtype = np.dtype(all_feature_types)
    features_data = np.empty(
        len(coordinates),
        dtype=dtype,
    )

    # apply all displacements and generate new arrays of features displacements
    for feature, dx, dy, dz in tqdm(features, desc="Filtering feature data by coordinates"):
        # Get the coordinates list
        feature_coordinates = coordinates.copy()
        feature_coordinates['x'] += dx
        feature_coordinates['y'] += dy
        feature_coordinates['z'] += dz
        coords = feature_coordinates

        # Filter coordinates into vals
        feature_data = feature_files[feature]
        vals = np.empty((len(coords), ), np.float64)
        for i, c in enumerate(coords):
            vals[i] = feature_data[tuple(c)]

        # Add feature data
        features_data[f'{feature_name}-{dx}-{dy}-{dz}'] = vals

    # Create dask array with a single feature per chunk for feature parallelism
    print(features_data)
    # features_data_da = [da.from_array(feature_data) for feature_data in features_data]

    # Create configs to run
    configurations = [fname for fname, _ in all_feature_types]
    configurations = [configurations[0]]

    print(
        f"Generated {len(configurations)} configurations"
    )

    client = Client(SCHEDULER_ADDRESS)
    # client.persist(training_data_da)
    client.upload_file("feature_sel.py")
    client.scatter(training_data, broadcast=True)
    client.scatter(features_data, broadcast=True)
    print(client)

    # Start workers
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

    f_iteration = 0
    tasks = [
        dask.delayed(single_feature_trial)(training_data, features_data, config, f_iteration)
        for config in configurations
    ]

    print('----------------------------------')
    print(training_data)
    print(features_data)
    results = dask.compute(*tasks)

    print(results)
    0/0
    
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
