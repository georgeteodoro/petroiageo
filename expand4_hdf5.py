import numpy as np
import time

import common
import hdf5_util


# pp is for showing the iteration progress, which can be enabled or disabled
def gen_expanded_points(d_h5, hypercube_shape, real_wells, it, it_str, pp):
    depth_len = hypercube_shape[2]
    ring = it

    t1 = time.time()

    if len(it_str) > 0:
        print(f'[gen_expanded_points]{it_str} Expanding points on ring {ring}')
    # Generate a list of points to be expanded
    well_id = 0
    for well in pp(real_wells):
        x_left = well[0] - ring
        x_right = well[0] + ring
        y_top = well[1] - ring
        y_bot = well[1] + ring

        # Conditions for points on each ring wall
        left_wall_cond = lambda d: (d['x'] == x_left) & (d['y'] <= y_bot) & (d[
            'y'] >= y_top)
        right_wall_cond = lambda d: (d['x'] == x_right) & (d['y'] <= y_bot) & (
            d['y'] >= y_top)
        top_wall_cond = lambda d: (d['y'] == y_top) & (d['x'] <= x_right) & (d[
            'x'] >= x_left)
        bot_wall_cond = lambda d: (d['y'] == y_bot) & (d['x'] <= x_right) & (d[
            'x'] >= x_left)

        # Update 'empty' values to 'expanded' if point is on any ring border
        hdf5_util.conditional_map_h5_all_clusters(
            d_h5, lambda d:
            (d['real'] == common.RealValues.empty) & (left_wall_cond(
                d) | right_wall_cond(d) | top_wall_cond(d) | bot_wall_cond(d)),
            [('real', common.RealValues.expanded), ('well_id', well_id)])
        hdf5_util.conditional_map_h5_all_clusters(
            d_h5, lambda d:
            (d['real'] == common.RealValues.canal) & (left_wall_cond(
                d) | right_wall_cond(d) | top_wall_cond(d) | bot_wall_cond(d)),
            [('real', common.RealValues.canal_expanded), ('well_id', well_id)])

        well_id = well_id + 1
