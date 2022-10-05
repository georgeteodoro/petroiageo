import numpy as np
import time
from tqdm import tqdm

import common
import hdf5_util


def gen_expanded_points(d_h5, hypercube_shape, real_wells, it):
    depth_len = hypercube_shape[2]
    ring = it

    t1 = time.time()

    to_expand = hdf5_util.fold_h5_all_clusters(
        d_h5,
        lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                        (d['real'] == common.RealValues.expanded)]), 0)

    print(f'[gen_expanded_points] Expanding points on ring {ring}')
    # Generate a list of points to be expanded
    for well in tqdm(real_wells):
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
            common.RealValues.expanded)
        hdf5_util.conditional_map_h5_all_clusters(
            d_h5, lambda d:
            (d['real'] == common.RealValues.canal) & (left_wall_cond(
                d) | right_wall_cond(d) | top_wall_cond(d) | bot_wall_cond(d)),
            common.RealValues.canal_expanded)

        to_expand = hdf5_util.fold_h5_all_clusters(
            d_h5,
            lambda d: len(d[(d['real'] == common.RealValues.canal_expanded) |
                            (d['real'] == common.RealValues.expanded)]), 0)
