import numpy as np
import dask.dataframe as dd
from math import prod

import common


def at_well(ddf, well_coords):
    return ddf[(ddf['x'] == well_coords[0]) & (ddf['y'] == well_coords[1])]


def get_canal_and_real_data(filename, hypercube_shape, real_points):

    total_points = prod(hypercube_shape)

    # Creates the DataFrame for the whole hypercube
    main_ddf = dd.from_array(np.full(total_points, common.RealValues.empty),
                             columns=['real'])
    main_ddf['phi'] = dd.from_array(np.zeros(total_points, dtype=float))
    main_ddf['phi_weights'] = dd.from_array(np.zeros(total_points,
                                                     dtype=float))

    # Setup index for main_ddf
    index = np.array(list(range(total_points)))
    main_ddf['index'] = dd.from_array(index)
    main_ddf = main_ddf.astype({'index': 'int64'})
    main_ddf = main_ddf.set_index('index')

    # Load known porosity canal points and add it to a dask DataFrame
    porosity_np = np.load(filename)
    canal_ddf = dd.from_array(porosity_np, columns=['x', 'y', 'z', 'phi'])

    # Create a dask DataFrame with only the real points
    real_ddf = at_well(canal_ddf, real_points[0])
    concat_list = []
    for well_coord in real_points[1:]:
        concat_list = concat_list + [at_well(canal_ddf, well_coord)]
    real_ddf = dd.concat([real_ddf] + concat_list)

    # Create 'index' for canal points with (x,y,z) (x is the highest dimension)
    canal_ddf['index'] = canal_ddf.z + canal_ddf.y * hypercube_shape[
        2] + canal_ddf.x * hypercube_shape[1] * hypercube_shape[2]
    canal_ddf = canal_ddf.astype({'index': 'int64'})
    canal_ddf = canal_ddf.set_index('index')

    # Create 'index' for real points with (x,y,z) (x is the highest dimension)
    real_ddf['index'] = real_ddf.z + real_ddf.y * hypercube_shape[
        2] + real_ddf.x * hypercube_shape[1] * hypercube_shape[2]
    real_ddf = real_ddf.astype({'index': 'int64'})
    real_ddf = real_ddf.set_index('index')

    # Make a mask of canal_ddf for the 'real' column
    canal_mask_ddf = canal_ddf.copy()
    canal_mask_ddf['phi'] = 0

    # Make a mask of real_ddf for the 'real' column
    real_mask_ddf = real_ddf.copy()
    real_mask_ddf['phi'] = 0

    # Set canal and real 'phi' values
    main_ddf['phi'] = main_ddf['phi'].add(canal_ddf['phi'], fill_value=0)

    # Set canal 'real' values
    main_ddf['real'] = main_ddf['real'].mul(canal_mask_ddf['phi'],
                                            fill_value=1)
    canal_mask_ddf['phi'] = common.RealValues.canal
    main_ddf['real'] = main_ddf['real'].add(canal_mask_ddf['phi'],
                                            fill_value=0)

    # Set real well points 'real' values
    main_ddf['real'] = main_ddf['real'].mul(real_mask_ddf['phi'], fill_value=1)
    real_mask_ddf['phi'] = common.RealValues.real
    main_ddf['real'] = main_ddf['real'].add(real_mask_ddf['phi'], fill_value=0)

    # # Filter real values from canal ddf and remove x,y,z coordinates
    # canal_ddf = canal_ddf.drop(columns=['x', 'y', 'z'])
    # real_mask_ddf['phi'] = 0
    # canal_ddf['phi'] = canal_ddf['phi'].mul(real_mask_ddf['phi'], fill_value=1)
    # canal_ddf = canal_ddf[canal_ddf['phi'] > 0]

    # return main_ddf.persist(), canal_ddf.persist()
    return main_ddf.persist()
