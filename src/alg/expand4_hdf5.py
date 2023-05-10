import numpy as np
from time import time

import common
import hdf5_util
import profiling


# pp: is for showing the iteration progress, which can be enabled or disabled.
# full_depth_chunks: whether the chunks for d_h5 includes the full depth, i.e.,
# there are no 2 chunks which are stacked upon each other. This allows faster
# checking for well/chunk overlaps
def gen_expanded_points(d_h5,
                        hypercube_shape,
                        real_wells,
                        it,
                        it_str,
                        pp,
                        config,
                        full_depth_chunks=True):
    depth_len = hypercube_shape[2]
    ring = it

    t1 = time()

    if len(it_str) > 0:
        print(f'[gen_expanded_points]{it_str} Expanding points on ring {ring}')
    # Generate a list of points to be expanded
    well_id = 0
    total_chunk_update_time = 0
    for well in pp(real_wells):
        well_x_left = well[0] - ring
        well_x_right = well[0] + ring
        well_y_top = well[1] - ring
        well_y_bot = well[1] + ring

        # Conditions for points on each ring wall
        left_wall_cond = lambda d: (d['x'] == well_x_left) & (d[
            'y'] <= well_y_bot) & (d['y'] >= well_y_top)
        right_wall_cond = lambda d: (d['x'] == well_x_right) & (d[
            'y'] <= well_y_bot) & (d['y'] >= well_y_top)
        top_wall_cond = lambda d: (d['y'] == well_y_top) & (d[
            'x'] <= well_x_right) & (d['x'] >= well_x_left)
        bot_wall_cond = lambda d: (d['y'] == well_y_bot) & (d[
            'x'] <= well_x_right) & (d['x'] >= well_x_left)

        # Used only for profiling
        n_chunk = -1
        ran_chunks = 0
        total_chunks = 0

        # Iterate on all chunks
        for chunk_slice in d_h5.iter_chunks():
            n_chunk += 1
            total_chunks += 1
            t2 = time()

            # Given the two rectangular regions: chunk and well, chunk have points
            # to be updated whenever chunk and well overlaps.

            chunk_x_left = chunk_slice[0].start
            chunk_x_right = chunk_slice[0].stop - 1
            chunk_y_top = chunk_slice[1].start
            chunk_y_bot = chunk_slice[1].stop - 1

            # chunk is used as a base to compare
            no_ovlp_x = (chunk_x_right < well_x_left) | (chunk_x_left >
                                                         well_x_right)
            no_ovlp_y = (chunk_y_bot < well_y_top) | (chunk_y_top > well_y_bot)

            if full_depth_chunks:
                # Ignore the current chunk if no overlapping is found
                if no_ovlp_x | no_ovlp_y:
                    continue
            else:
                print('[expand4_hdf5] Not using full_depth_chunks=True')
                raise NotImplementedError

            ran_chunks += 1  # Used only for profiling

            # Calculate whether the current chunk has any points to update/expand
            # print(chunk_slice)

            # Update 'empty' values to 'expanded' if point is on any ring border
            hdf5_util.conditional_map_h5_chunk(
                d_h5, lambda d: (d['real'] == common.RealValues.empty) &
                (left_wall_cond(d) | right_wall_cond(d) | top_wall_cond(d) |
                 bot_wall_cond(d)), [('real', common.RealValues.expanded),
                                     ('well_id', well_id)], chunk_slice)
            hdf5_util.conditional_map_h5_chunk(
                d_h5, lambda d: (d['real'] == common.RealValues.canal) &
                (left_wall_cond(d) | right_wall_cond(d) | top_wall_cond(d) |
                 bot_wall_cond(d)),
                [('real', common.RealValues.canal_expanded),
                 ('well_id', well_id)], chunk_slice)

            t3 = time()
            total_chunk_update_time += t3 - t2
            profiling.prof_expand_chunk_time(it, n_chunk, t3 - t2, config)

        well_id = well_id + 1

    profiling.prof_expand_chunks_time(it, total_chunk_update_time, config)
    profiling.prof_expand_chunks_ran(it, ran_chunks, total_chunks, config)
