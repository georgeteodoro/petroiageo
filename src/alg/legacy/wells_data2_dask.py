import numpy as np
import dask.dataframe as dd
from math import prod

import common


def at_well(ddf, well_coords):
    return ddf[(ddf["x"] == well_coords[0]) & (ddf["y"] == well_coords[1])]


def get_canal_and_real_data(filename, hypercube_shape, real_points):
    total_points = prod(hypercube_shape)
    chunksize = 10_000_000

    # Creates the DataFrame for the whole hypercube
    print("Creating DataFrame for the whole hypercube")
    main_ddf = dd.from_array(
        np.full(total_points, common.RealValues.empty),
        columns=["real"],
        chunksize=chunksize,
    )

    # Create coordinates values
    print("Creating coordinates")
    xs_np = np.zeros(prod(hypercube_shape), dtype=np.int64)
    ys_np = np.zeros(prod(hypercube_shape), dtype=np.int64)
    zs_np = np.zeros(prod(hypercube_shape), dtype=np.int64)
    ii = 0
    for i in range(hypercube_shape[0]):
        for j in range(hypercube_shape[1]):
            for k in range(hypercube_shape[2]):
                xs_np[ii] = int(i)
                ys_np[ii] = int(j)
                zs_np[ii] = int(k)
                ii = ii + 1

    print("Setting coordinates")
    main_ddf["x"] = dd.from_array(xs_np, chunksize=chunksize)
    main_ddf["y"] = dd.from_array(ys_np, chunksize=chunksize)
    main_ddf["z"] = dd.from_array(zs_np, chunksize=chunksize)

    main_ddf = main_ddf[["x", "y", "z", "real"]]

    main_ddf = main_ddf.persist()
    print("")
    del xs_np
    del ys_np
    del zs_np

    main_ddf["well_id"] = dd.from_array(
        np.full(total_points, -1, dtype=np.int64), chunksize=chunksize
    )

    main_ddf["phi"] = dd.from_array(
        np.zeros(total_points, dtype=float), chunksize=chunksize
    )
    main_ddf["phi_weights"] = dd.from_array(
        np.zeros(total_points, dtype=float), chunksize=chunksize
    )
    main_ddf = main_ddf.persist()

    # Setup index for main_ddf
    print("Setup index for main_ddf")
    index = np.array(list(range(total_points)))
    main_ddf["index"] = dd.from_array(index, chunksize=chunksize)
    main_ddf = main_ddf.astype({"index": "int64"})
    main_ddf = main_ddf.set_index("index")
    main_ddf = main_ddf.persist()

    # Load known porosity canal points and add it to a dask DataFrame
    print("Loading canal points")
    porosity_np = np.load(filename)
    canal_ddf = dd.from_array(
        porosity_np, columns=["x", "y", "z", "phi"], chunksize=chunksize
    )

    # Create a dask DataFrame with only the real points
    print("Loading real points")
    real_ddf = at_well(canal_ddf, real_points[0])
    well_id = 0
    real_ddf["well_id"] = well_id
    concat_list = []
    for well_coord in real_points[1:]:
        well_rows_ddf = at_well(canal_ddf, well_coord)
        well_id = well_id + 1
        well_rows_ddf["well_id"] = well_id
        concat_list = concat_list + [well_rows_ddf]
    real_ddf = dd.concat([real_ddf] + concat_list)

    # Create 'index' for canal points with (x,y,z) (x is the highest dimension)
    print("Creating indices")
    canal_ddf["index"] = (
        canal_ddf.z
        + canal_ddf.y * hypercube_shape[2]
        + canal_ddf.x * hypercube_shape[1] * hypercube_shape[2]
    )
    canal_ddf = canal_ddf.astype({"index": "int64"})
    canal_ddf = canal_ddf.set_index("index")

    # Create 'index' for real points with (x,y,z) (x is the highest dimension)
    real_ddf["index"] = (
        real_ddf.z
        + real_ddf.y * hypercube_shape[2]
        + real_ddf.x * hypercube_shape[1] * hypercube_shape[2]
    )
    real_ddf = real_ddf.astype({"index": "int64"})
    real_ddf = real_ddf.set_index("index")

    canal_ddf = canal_ddf.persist()
    real_ddf = real_ddf.persist()

    # Make a mask of canal_ddf for the 'real' column
    canal_mask_ddf = canal_ddf.copy()
    canal_mask_ddf["phi"] = 0

    # Make a mask of real_ddf for the 'real' column
    real_mask_ddf = real_ddf.copy()
    real_mask_ddf["phi"] = 0

    canal_mask_ddf = canal_mask_ddf.persist()
    real_mask_ddf = real_mask_ddf.persist()

    # Set canal and real 'phi' values
    print("Setting main_ddf")
    main_ddf["phi"] = main_ddf["phi"].add(canal_ddf["phi"], fill_value=0)
    main_ddf = main_ddf.persist()

    # Set canal 'real' values
    main_ddf["real"] = main_ddf["real"].mul(canal_mask_ddf["phi"], fill_value=1)
    canal_mask_ddf["phi"] = common.RealValues.canal
    main_ddf["real"] = main_ddf["real"].add(canal_mask_ddf["phi"], fill_value=0)

    # Set real well points 'real' values
    main_ddf["real"] = main_ddf["real"].mul(real_mask_ddf["phi"], fill_value=1)
    real_mask_ddf["phi"] = common.RealValues.real
    main_ddf["real"] = main_ddf["real"].add(real_mask_ddf["phi"], fill_value=0)

    # Set 'well_id' values
    main_ddf["well_id"] = real_ddf["well_id"]
    main_ddf["well_id"] = main_ddf["well_id"].fillna(-1)
    main_ddf = main_ddf.persist()

    # # Filter real values from canal ddf and remove x,y,z coordinates
    # canal_ddf = canal_ddf.drop(columns=['x', 'y', 'z'])
    # real_mask_ddf['phi'] = 0
    # canal_ddf['phi'] = canal_ddf['phi'].mul(real_mask_ddf['phi'], fill_value=1)
    # canal_ddf = canal_ddf[canal_ddf['phi'] > 0]

    # return main_ddf.persist(), canal_ddf.persist()
    main_ddf = main_ddf.repartition(npartitions=8)
    return main_ddf.persist()


if __name__ == "__main__":
    real_wells = [
        (134, 227),
        (146, 500),
        (167, 186),
        (174, 365),
        (200, 102),
        (236, 113),
        (250, 315),
        (287, 242),
        (230, 194),
        (344, 276),
    ]
    hypercube_shape = np.load(f"./dados/NEAR.npy").shape

    main_ddf = get_canal_and_real_data(
        "./dados/porosity-canal.npy", hypercube_shape, real_wells
    )

    print("Writing data")
    print(main_ddf)
    print(main_ddf.head())
    main_ddf.to_parquet("./dados/initial_main_ddf.parquet")
    # print(features.compute())
