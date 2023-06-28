import numpy as np
import time
import dask.dataframe as dd

import common


def get_coord(x, y, z, hypercube_shape):
    return (
        x * hypercube_shape[2] * hypercube_shape[1] + y * hypercube_shape[2] + z
    )


def gen_expanded_points(main_ddf, hypercube_shape, real_wells, it):
    depth_len = hypercube_shape[2]

    # Set distance ring to be generated
    ring = it

    t1 = time.time()

    # Generate a list of points to be expanded
    expanded_indices = []
    for well in real_wells:
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                # Only add ring frontier points
                if abs(i) == ring or abs(j) == ring:
                    expanded_indices = expanded_indices + list(
                        range(
                            get_coord(
                                well[0] + i, well[1] + j, 0, hypercube_shape
                            ),
                            get_coord(
                                well[0] + i,
                                well[1] + j,
                                depth_len,
                                hypercube_shape,
                            ),
                        )
                    )

    # Get the list of points to be expanded
    to_expand_ddf = main_ddf.loc[expanded_indices]

    # Update empty points to expanded
    empty_to_expand_ddf = to_expand_ddf[
        to_expand_ddf["real"] == common.RealValues.empty
    ]
    empty_to_expand_ddf["real"] = 0
    main_ddf["real"] = main_ddf["real"].mul(
        empty_to_expand_ddf["real"], fill_value=int(1)
    )
    empty_to_expand_ddf["real"] = common.RealValues.expanded
    main_ddf["real"] = main_ddf["real"].add(
        empty_to_expand_ddf["real"], fill_value=int(0)
    )

    # Update canal points to canal_expanded
    canal_to_expand_ddf = to_expand_ddf[
        to_expand_ddf["real"] == common.RealValues.canal
    ]
    canal_to_expand_ddf["real"] = 0
    main_ddf["real"] = main_ddf["real"].mul(
        canal_to_expand_ddf["real"], fill_value=int(1)
    )
    canal_to_expand_ddf["real"] = common.RealValues.canal_expanded
    main_ddf["real"] = main_ddf["real"].add(
        canal_to_expand_ddf["real"], fill_value=int(0)
    )

    main_ddf = main_ddf.persist()

    t2 = time.time()

    return main_ddf
