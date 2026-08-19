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

#BASE_PATH="/home/will/git/petroiageo"
BASE_PATH="/snfs2/willianjunior/git/petroiageo"

SCHEDULER_ADDRESS = "tcp://127.0.0.1:8786"
# DATASET_PATH = Path(BASE_PATH + "/data/POV/porosity_data.h5")
DATASET_PATH = BASE_PATH + "/data/POV/raw/porosity-canal.txt"
POROSITY_DSET_NAME = "p"
FEAT_DSET_NAME = "f"
DISP_WINDOW_X = 1
DISP_WINDOW_Y = DISP_WINDOW_X
DISP_WINDOW_Z = DISP_WINDOW_X
FEATURES_PATH = BASE_PATH + "/data/POV/features"
#FEATURES_LIST = ['FAR', 'FAR2']
FEATURES_LIST = ['FAR']

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
# 3. Actual task
# ============================================================

def single_feature_trial(training_data, features_data, feature_name, f_iteration):
    feature = features_data[feature_name]
    training_data[f'f_{f_iteration}'] = feature
    rmse, mae = test_new_feature(training_data, f_iteration)
    print(f"[single_feature_trial][{worker.id}][f_it{f_iteration}] done {feature_name}: {rmse}, {mae}")

    return feature_name, rmse, mae



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
    data_filtered = np.empty(
        len(raw),
        dtype=dtype,
    )
    data["x"] = raw[:, 0]
    data["y"] = raw[:, 1]
    data["z"] = raw[:, 2]
    data["phi"] = raw[:, 3]
    data["well_id"] = -1
    data["real"] = RealValues.real

    # Filter training data for only the well
    filtered_size = 0
    for w_x, w_y in REAL_WELLS:
        for x in range(w_x-DISP_WINDOW_X, w_x+DISP_WINDOW_X+1):
            for y in range(w_y-DISP_WINDOW_Y, w_y+DISP_WINDOW_Y+1):
                filt = data[(data['x'] == x) & (data['y'] == y)]
                data_filtered[filtered_size:filtered_size+len(filt)] = filt
                filtered_size += len(filt)
    
    
    data_filtered.resize(filtered_size, refcheck=False)
    print(f"data_filtered: {data_filtered[:10]}...")
    print(f"filtered training data: {data_filtered.shape}")

    real_wells_dict = {}
    for i, w in enumerate(REAL_WELLS):
        real_wells_dict[w] = i

    print(f"real_wells_dict: {real_wells_dict}")

    for i, d in enumerate(data_filtered):
        x, y = int(d['x']), int(d['y'])
        if (x, y) in real_wells_dict:
            data_filtered[i]['well_id'] = real_wells_dict[(x,y)]

    training_data = data_filtered

    # get coordinates of current points
    coordinates = training_data[['x','y','z']]
    print(f"coordinates: {coordinates[:10]}...")

    # Prepare the features
    all_feature_types = []
    mpi_kwargs = {}
    features = []
    feature_files = {}
    for feature_name in FEATURES_LIST:
        feature_file = np.load(FEATURES_PATH + "/" + feature_name + ".npy")
        feature_files[feature_name] = feature_file
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
    print(f"features: {features_data.shape}")

    # Create configs to run
    configurations = [fname for fname, _ in all_feature_types]

    print(
        f"Generated {len(configurations)} configurations"
    )

    client = Client(SCHEDULER_ADDRESS)
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
    with open("training.log", "w") as f:
        #f.write("".join(f"{row}\n" for row in training_data))
        f.writelines(map(lambda x: repr(x) + "\n", training_data))
        #for row in tqdm(training_data):
        #    log_file.write(f"{row}\n")

    results = dask.compute(*tasks)

    print(results)
    
    return results


if __name__ == "__main__":
    main()
