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


def single_feature_trial(config):
    from dask.distributed import get_worker
    feature_name, dx, dy, dz = config

    worker = get_worker()
    print(f"[single_feature_trial][{worker.id}] testing {config}")
    print(worker.dataset_h5)
    training_data = worker.dataset_h5
    training_data = training_data[training_data['real'] != RealValues.empty]
    print(f"[single_feature_trial][{worker.id}] done copying")
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

def shift_array_3d(arr, dx=0, dy=0, dz=0, default=0):
    """
    Shift a 3D NumPy array by (dx, dy, dz).

    Coordinates are interpreted as (x, y, z), while the array
    axes are (z, y, x).

    Integer shifts are required.

    Values shifted outside the array bounds are discarded.
    Newly exposed positions are filled with `default`.
    """

    if arr.ndim != 3:
        raise ValueError("arr must be a 3D array")

    if not all(float(v).is_integer() for v in (dx, dy, dz)):
        raise ValueError("dx, dy, dz must be integers")

    dx, dy, dz = int(dx), int(dy), int(dz)

    result = np.full_like(arr, default)

    nz, ny, nx = arr.shape

    # Source ranges
    src_x0 = max(0, -dx)
    src_x1 = min(nx, nx - dx)

    src_y0 = max(0, -dy)
    src_y1 = min(ny, ny - dy)

    src_z0 = max(0, -dz)
    src_z1 = min(nz, nz - dz)

    # Destination ranges
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    dst_z0 = max(0, dz)
    dst_z1 = dst_z0 + (src_z1 - src_z0)

    result[
        dst_z0:dst_z1,
        dst_y0:dst_y1,
        dst_x0:dst_x1,
    ] = arr[
        src_z0:src_z1,
        src_y0:src_y1,
        src_x0:src_x1,
    ]

    return result

def main():

    # client = Client(SCHEDULER_ADDRESS)
    # print(client)

    #client.register_plugin(
    #    DatasetPlugin(str(DATASET_PATH))
    #)

    # load porosity list .txt
    # have x, y, z, real, well, phi cols

    dtype = np.dtype([
        ("x", np.int32),
        ("y", np.int32),
        ("z", np.int32),
        ("phi", np.float32),
        ("real", np.int32),
    ])
    raw = np.loadtxt(DATASET_PATH)
    data = np.empty(
        len(raw),
        dtype=dtype,
    )
    data["x"] = raw[:, 0]
    data["y"] = raw[:, 1]
    data["z"] = raw[:, 2]
    data["phi"] = raw[:, 3]
    data["real"] = 5

    training_data = data

    # get coordinates of current points
    coordinates = training_data[['x','y','z']]

    print(training_data)
    print(coordinates)

    mpi_kwargs = {}
    features = []
    feature_files = {}
    for feature in FEATURES_LIST:
        feature_file = h5py.File(FEATURES_PATH + "/" + feature + ".h5",
                                'r', **mpi_kwargs)
        feature_files[feature] = feature_file[FEAT_DSET_NAME]
        for dx in range(-DISP_WINDOW_X, DISP_WINDOW_X + 1):
            for dy in range(-DISP_WINDOW_Y, DISP_WINDOW_Y + 1):
                for dz in range(-DISP_WINDOW_Z, DISP_WINDOW_Z + 1):
                    features.append((feature, dx, dy, dz))

    print(features)
    print(feature_files)

    # apply all displacements and generate new arrays of features displacements
    all_features_data = []
    for feature, dx, dy, dz in features:
        feature_coordinates = coordinates.copy()
        feature_coordinates['x'] += dx
        feature_coordinates['y'] += dy
        feature_coordinates['z'] += dz

        feature_data = feature_files[feature]
        xyz = feature_data[["x", "y", "z"]][:]
        mask = np.isin(xyz, feature_coordinates)
        # Read the matching complete rows
        # return h5_dataset[:][mask]

        feature_data[FEAT_DSET_NAME] = feature_file[feature_coordinates]

        all_features_data.append(feature_data)

    print(all_features_data)
    0/0

    # concatenate all columns
    training_data_da = da.concatenate([training_data] + all_features_data, axis=1)













    mpi_kwargs = {}
    dataset_h5 = h5py.File(DATASET_PATH,
                                   'r', **mpi_kwargs)
    dataset_h5 = worker.dataset_h5[POROSITY_DSET_NAME]
    
    all_datasets = [dataset_h5]


    for feature in FEATURES_LIST:
        feature_dataset = h5py.File(DATASET_PATH,
                                   'r', **mpi_kwargs)
        feature_dataset = feature_dataset[FEATURE_DSET_NAME]
        for i in range(-DISP_WINDOW_X, DISP_WINDOW_X + 1):
            for j in range(-DISP_WINDOW_Y, DISP_WINDOW_Y + 1):
                for k in range(-DISP_WINDOW_Z, DISP_WINDOW_Z + 1):
                    feature = shift_array_3d(np.asarray(feature_dataset), i, j, k)
                    all_datasets.append(feature)


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
