import h5py
from time import time

from inverted_learning_interface import AbstractExpandAlg
import config_parser
import hdf5_util
import common
import profiling
from typing import List, Tuple


class H5ExpandAlg(AbstractExpandAlg):
    """
    Expansion algorithm for HDF5 files and MPI.
    Although MPI support is there, it is possible to run it without 'mpirun'.
    Expanded points are updated in-place on the porosity hypercube.
    """

    def __init__(self, config: config_parser.Config):
        self._config = config

        # Compatibility flags:
        super().__init__()
        self._using_h5 = True

    def expand_points(self, porosity_data_h5: h5py.Dataset, it: int):
        """
        Expand training wells
        """
        # Retrieve config parameters
        wells_coords = self._config.train_wells_coords
        comm = self._config.get_param('mpi_global_comm')
        rank = self._config.get_param('mpi_rank')
        should_update = self._config.get_param('mpi_should_update_local')
        full_depth_chunks = self._config.get_param('full_depth_chunks')

        # Only one process per node is required to update
        if should_update:
            self._expand(porosity_data_h5, it, wells_coords, full_depth_chunks)

        else:
            my_rank = rank
            print(f"[main][{it}][R{my_rank}] waiting points expansion")

        comm.Barrier()

    def _expand(self, porosity_data_h5: h5py.Dataset, it: int,
                wells_coords: List[Tuple[int, int]], full_depth_chunks: bool):
        all_wells_coords = self._config.wells_as_simple_list

        print(f"[gen_expanded_points][it{it}] "\
                  f"Hypercube shape: {porosity_data_h5.shape}")

        # ring start at 1
        start_ring, end_ring = self._config.ring_range_to_expand(it)

        total_chunk_update_time = 0
        # Used only for profiling
        n_chunk = -1
        ran_chunks = 0
        total_chunks = 0

        # We expand ring by ring so we dont mark many points belonging
        # to a well right away and influence the expansion of other
        # wells
        for ring in range(start_ring, end_ring + 1):
            print(f"[gen_expanded_points][it{it}] "\
                    f"Expanding points on ring {ring}")

            # Generate a list of points to be expanded

            # TODO: add progress bar later
            for well_coords in wells_coords:
                well_id = all_wells_coords.index(well_coords)
                well_x_left = well_coords[0] - ring
                well_x_right = well_coords[0] + ring
                well_y_top = well_coords[1] - ring
                well_y_bot = well_coords[1] + ring

                # Conditions for points on each ring wall
                left_wall_cond = (lambda d: (d['x'] == well_x_left)
                                  & (d['y'] <= well_y_bot)
                                  & (d['y'] >= well_y_top))
                right_wall_cond = (lambda d: (d['x'] == well_x_right)
                                   & (d['y'] <= well_y_bot)
                                   & (d['y'] >= well_y_top))
                top_wall_cond = (lambda d: (d['y'] == well_y_top)
                                 & (d['x'] <= well_x_right)
                                 & (d['x'] >= well_x_left))
                bot_wall_cond = (lambda d: (d['y'] == well_y_bot)
                                 & (d['x'] <= well_x_right)
                                 & (d['x'] >= well_x_left))

                # Iterate on all chunks
                for chunk_slice in porosity_data_h5.iter_chunks():
                    n_chunk += 1
                    total_chunks += 1
                    t2 = time()

                    # Calculate whether the current chunk has any points to
                    # update/expand

                    # Given the two rectangular regions: chunk and well,
                    # chunk have points to be updated whenever chunk
                    # and well overlaps.

                    chunk_x_left = chunk_slice[0].start
                    chunk_x_right = chunk_slice[0].stop - 1
                    chunk_y_top = chunk_slice[1].start
                    chunk_y_bot = chunk_slice[1].stop - 1

                    # chunk is used as a base to compare
                    no_ovlp_x = (chunk_x_right
                                 < well_x_left) | (chunk_x_left > well_x_right)
                    no_ovlp_y = (chunk_y_bot < well_y_top) | (chunk_y_top
                                                              > well_y_bot)

                    # full_depth_chunks: whether the chunks for
                    # porosity_data_h5 includes the full depth, i.e.,
                    # there are no 2 chunks which are stacked upon each other.
                    # This allows faster checking for well/chunk overlaps
                    if full_depth_chunks:
                        # Ignore the current chunk if no overlapping is found
                        if no_ovlp_x | no_ovlp_y:
                            continue
                    else:
                        print(
                            "[expand4_hdf5] Not using full_depth_chunks=True")
                        raise NotImplementedError

                    ran_chunks += 1  # Used only for profiling

                    within_chunk_cond = (lambda d: (d['x'] >= chunk_x_left) &
                                         (d['x'] <= chunk_x_right) &
                                         (d['y'] >= chunk_y_top) &
                                         (d['y'] <= chunk_y_bot))

                    # Update 'empty' values to 'expanded' if point is
                    # on any ring border and if they are present on
                    # this chunk
                    local_cond = lambda d: within_chunk_cond(d) & (
                        left_wall_cond(d)
                        | right_wall_cond(d)
                        | top_wall_cond(d)
                        | bot_wall_cond(d))

                    hdf5_util.conditional_map_h5_chunk(
                        porosity_data_h5,
                        lambda d:
                        (d['real'] == common.RealValues.empty) & local_cond(d),
                        [('well_id', well_id),
                         ('real', common.RealValues.expanded), ('ring', ring)],
                        chunk_slice,
                    )

                    hdf5_util.conditional_map_h5_chunk(
                        porosity_data_h5,
                        lambda d: (d['real'] == common.RealValues.canal)
                        & local_cond(d),
                        [('well_id', well_id),
                         ('real', common.RealValues.expanded), ('ring', ring)],
                        chunk_slice,
                    )

                    t3 = time()
                    total_chunk_update_time += t3 - t2
                    profiling.prof_expand_chunk_time(it, well_id, n_chunk,
                                                     t3 - t2)

        profiling.prof_expand_chunks_time(it, total_chunk_update_time,
                                          self._config)

        profiling.prof_expand_chunks_ran(it, ran_chunks, total_chunks,
                                         self._config)

    def expand_points_improved_locality(self, porosity_data_h5: h5py.Dataset,
                                        it: int):
        # Retrieve config parameters
        wells_coords = self._config.wells_as_simple_list
        comm = self._config.get_param('mpi_global_comm')
        rank = self._config.get_param('mpi_rank')
        should_update = self._config.get_param('mpi_should_update_local')
        full_depth_chunks = self._config.get_param('full_depth_chunks')
        window_size = self._config.get_param('window')

        # Only one process per node is required to update
        if should_update:
            ring = it

            t1 = time()

            print(f"[gen_expanded_points][it{it}] "\
                  f"Hypercube shape: {porosity_data_h5.shape}")

            print(f"[gen_expanded_points][it{it}] "\
                  f"Expanding points on ring {ring}")

            total_chunk_update_time = 0

            # TODO: add progress bar later
            # Iterate on all chunks
            for chunk_slice in porosity_data_h5.iter_chunks():

                chunk_x_left = chunk_slice[0].start
                chunk_x_right = chunk_slice[0].stop - 1
                chunk_y_top = chunk_slice[1].start
                chunk_y_bot = chunk_slice[1].stop - 1

                # Create a list of wells which have
                # points within the current chunk
                wells_to_update = []
                for well_id, well in enumerate(wells_coords):
                    # Calculate the coordinates of the current well-ring
                    well_ring_x_left = well[0] - ring
                    well_ring_x_right = well[0] + ring
                    well_ring_y_top = well[1] - ring
                    well_ring_y_bot = well[1] + ring

                    # Check if any point in the well-ring is within the chunk
                    ovlp_x = (chunk_x_right >= well_ring_x_left) & (
                        chunk_x_left <= well_ring_x_right)
                    ovlp_y = (chunk_y_bot >= well_ring_y_top) & (
                        chunk_y_top <= well_ring_y_bot)

                    # If the chunk region is withing a well-ring the overlap
                    # conditions are true, but there is no overlap.
                    well_ring_not_bigger = (
                        well_ring_x_left >= chunk_x_left | well_ring_x_right <=
                        chunk_x_right | well_ring_y_top >=
                        chunk_y_top | well_ring_y_bot <= chunk_y_bot)

                    # full_depth_chunks: whether the chunks for
                    # porosity_data_h5 includes the full depth, i.e.,
                    # there are no 2 chunks which are stacked upon each other.
                    # This allows faster checking for well/chunk overlaps
                    if full_depth_chunks:
                        if (ovlp_x & ovlp_y) & well_ring_not_bigger:
                            wells_to_update.append(well_id)
                    else:
                        print(
                            "[expand4_hdf5] Not using full_depth_chunks=True")
                        raise NotImplementedError

                # Update the values of each well point withing the current chunk
                for well_id in wells_to_update:
                    # Calculate the coordinates of the current well-ring
                    well_ring_x_left = wells_coords[well_id][0] - ring
                    well_ring_x_right = wells_coords[well_id][0] + ring
                    well_ring_y_top = wells_coords[well_id][1] - ring
                    well_ring_y_bot = wells_coords[well_id][1] + ring

                    # Conditions for points on each ring wall
                    left_wall_cond = (lambda d: (d['x'] == well_ring_x_left)
                                      & (d['y'] <= well_ring_y_bot)
                                      & (d['y'] >= well_ring_y_top))
                    right_wall_cond = (lambda d: (d['x'] == well_ring_x_right)
                                       & (d['y'] <= well_ring_y_bot)
                                       & (d['y'] >= well_ring_y_top))
                    top_wall_cond = (lambda d: (d['y'] == well_ring_y_top)
                                     & (d['x'] <= well_ring_x_right)
                                     & (d['x'] >= well_ring_x_left))
                    bot_wall_cond = (lambda d: (d['y'] == well_ring_y_bot)
                                     & (d['x'] <= well_ring_x_right)
                                     & (d['x'] >= well_ring_x_left))

                    # Update 'empty' values to 'expanded' if point is
                    # on any ring border
                    hdf5_util.conditional_map_h5_chunk(
                        porosity_data_h5,
                        lambda d: (d['real'] == common.RealValues.empty)
                        & (left_wall_cond(d)
                           | right_wall_cond(d)
                           | top_wall_cond(d)
                           | bot_wall_cond(d)),
                        [
                            ('real', common.RealValues.expanded),
                            ('well_id', well_id),
                        ],
                        chunk_slice,
                    )
                    hdf5_util.conditional_map_h5_chunk(
                        porosity_data_h5,
                        lambda d: (d['real'] == common.RealValues.canal)
                        & (left_wall_cond(d)
                           | right_wall_cond(d)
                           | top_wall_cond(d)
                           | bot_wall_cond(d)),
                        [
                            ('real', common.RealValues.canal_expanded),
                            ('well_id', well_id),
                        ],
                        chunk_slice,
                    )

        else:
            my_rank = rank
            print(f"[main][{it}][R{my_rank}] waiting points expansion")

        comm.Barrier()

    def _single_compatible(self, to_compare):
        # Check if to_compare have h5 support
        compatible = to_compare._using_h5

        return compatible
