from math import ceil, prod
import numpy as np
from typing import Tuple
import h5py
from time import time
import argparse
import sys
import common
from random import random
from tqdm import tqdm


# Returns whether two slices have any overlapping.
# Includes the case of 1 within the other
def slice_overlaps_xy(s1, s2):
    no_ovlp_x = (s1[0].stop < s2[0].start) | (s1[0].start > s2[0].stop)
    no_ovlp_y = (s1[1].stop < s2[1].start) | (s1[1].start > s2[1].stop)

    return not (no_ovlp_x or no_ovlp_y)


def fold_h5_all_clusters(d_h5, f, out_0):
    """
    Perform a fold on a clustered h5 object, using the least amount
    of memory.
    """
    chunks = d_h5.chunks
    x_shape = d_h5.shape[0]
    y_shape = d_h5.shape[1]
    z_shape = d_h5.shape[2]

    out = out_0

    # Iterate on all coordinates
    for c_x in range(ceil(x_shape / chunks[0])):
        x_i = c_x * chunks[0]
        x_o = min((c_x + 1) * chunks[0], x_shape)
        for c_y in range(ceil(y_shape / chunks[1])):
            y_i = c_y * chunks[1]
            y_o = min((c_y + 1) * chunks[1], y_shape)
            for c_z in range(ceil(z_shape / chunks[2])):
                z_i = c_z * chunks[2]
                z_o = min((c_z + 1) * chunks[2], z_shape)

                # Read numpy chunk
                d_np = d_h5[x_i:x_o, y_i:y_o, z_i:z_o]

                # Append result of function and fold on out
                out = out + f(d_np)

    return out


def conditional_map_h5_all_clusters(d_h5, cond_f, column_val_list):
    """
    Perform a conditional update on a clustered h5 object, using the least amount
    of memory. All rows from the given 'column' for which the condition is True
    are updated to 'val'.
    """
    chunks = d_h5.chunks
    x_shape = d_h5.shape[0]
    y_shape = d_h5.shape[1]
    z_shape = d_h5.shape[2]

    # Iterate on all coordinates
    for c_x in range(ceil(x_shape / chunks[0])):
        x_i = c_x * chunks[0]
        x_o = min((c_x + 1) * chunks[0], x_shape)
        for c_y in range(ceil(y_shape / chunks[1])):
            y_i = c_y * chunks[1]
            y_o = min((c_y + 1) * chunks[1], y_shape)
            for c_z in range(ceil(z_shape / chunks[2])):
                z_i = c_z * chunks[2]
                z_o = min((c_z + 1) * chunks[2], z_shape)

                # Read numpy chunk
                d_np = d_h5[x_i:x_o, y_i:y_o, z_i:z_o]
                # print(f'=======points to update: {len(d_np[cond_f(d_np)])}')

                # Update values of each column on condition
                for column, val in column_val_list:
                    d_np[column] = np.where(cond_f(d_np), val, d_np[column])

                # Forward values to hdf5 file
                d_h5[x_i:x_o, y_i:y_o, z_i:z_o] = d_np


def conditional_map_h5_chunk(d_h5, cond_f, column_val_list, chunk_slice):
    # Read numpy chunk
    chunk_np = d_h5[chunk_slice]

    # Update values of each column on condition
    cond = cond_f(chunk_np)
    for column, val in column_val_list:
        chunk_np[column] = np.where(cond, val, chunk_np[column])

    # Forward values to hdf5 file
    d_h5[chunk_slice] = chunk_np


class HDFMultiColList:
    """
    Out-of-core structure to access hdf5 a dataset with chunking.
    Copying this object does not copy its data.
    Both train and validation data are inside
    Able to add new features columns on the fly as well as change a given column
    """

    def __init__(self, cur_h5_dset: h5py.Dataset):
        """
        cur_h5_dset must be 1D
        """
        # This cur_h5_dset holds the hdf5 data
        self.cur_h5_dset = cur_h5_dset

        # should be a list of strings
        # self.all_features_and_coords = base_features
        self.all_features = []
        self.last_col = -1

        if self.cur_h5_dset.chunks:
            self.chunk_size = self.cur_h5_dset.chunks[0]
            self.n_chunks = int(np.ceil(cur_h5_dset.size / self.chunk_size))
        else:
            self.chunk_size = None
            self.n_chunks = -1

        # if sampling_config is not None:
        #     self.sampling_window = sampling_config['its_window_size']
        #     self.sampling_max_points = sampling_config['max_points']
        # else:
        #     self.sampling_window = None
        #     self.sampling_max_points = None

    # Updates the last column with new values from a generator
    # Overwrites the previous values on this column
    def update_last_col_chunk(self, feature_gen):
        """
        Updates the last column with new values from a generator
        Overwrites the previous values on this column
        """
        f_str = f"f{self.last_col}"

        seismic_read_time = 0
        TD_write_time = 0

        for f_slice, f_vals in feature_gen:
            t0 = time()
            seismic_data = np.fromiter(f_vals, np.float64)
            t1 = time()
            self.cur_h5_dset[f_str, f_slice] = seismic_data
            t2 = time()
            seismic_read_time += t1 - t0
            TD_write_time += t2 - t1

        print(f"[insert_filtered_feature] IO_SEISMIC_READ: "
              f"{seismic_read_time}")
        print(f"[insert_filtered_feature] IO_TD_WRITE: {TD_write_time}")

    def update_last_col(self, feature_gen):
        """
        Updates the last column with new values from a generator
        Overwrites the previous values on this column
        """
        f_str = f"f{self.last_col}"

        t0 = time()
        seismic_data = np.fromiter(feature_gen, np.float64)
        t1 = time()
        self.cur_h5_dset[f_str] = seismic_data
        t2 = time()

        print(f"[insert_filtered_feature] "
              f"IO_SEISMIC_READ: {t1-t0}")
        print(f"[insert_filtered_feature] "
              f"IO_TD_WRITE: {t2-t1}")

    def add_new_col(self):
        """
        Setup a new empty last column, thus committing the current
        last column
        """
        self.last_col = self.last_col + 1
        f_str = f"f{self.last_col}"
        # self.all_features_and_coords.append(f_str)
        self.all_features.append(f_str)

    def get_data_from_well(self, well_id) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns all data, input (X) and output (y), from a given well.
        y is corresponds to the 'phi' column.
        X corresponds to all cols - 'phi'.
        """

        if not self.cur_h5_dset.chunks:
            # Filter all data which has the given well_id
            well_data = self.cur_h5_dset[self.cur_h5_dset["well_id"] ==
                                         well_id]

            X = well_data[self.all_features]
            y = well_data["phi"]
        else:
            X_val_list = []
            y_val_list = []

            for cur_slice in self.cur_h5_dset.iter_chunks():
                # Load a single hypercube chunk
                cur_chunk = self.cur_h5_dset[cur_slice]

                # Filter all data which has the given well_id
                well_data = cur_chunk[cur_chunk["well_id"] == well_id]

                # Append results to output lists
                X_val_list.append(well_data[self.all_features])
                y_val_list.append(well_data["phi"])

            X = np.concatenate(X_val_list).reshape(-1)
            y = np.concatenate(y_val_list).reshape(-1)

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        X = np.array(X.tolist())
        y = np.array(y.tolist())

        return X, y

    def get_data_not_in_well(self, chunk=-1, well_id=-1):
        """
        Return a single chunk of data which is not related to well_id.
        If the chunk is not passed (-1), returns the whole data.
        No guarantees are made about the size of the output.
        """
        cur_chunk = self._get_target_chunk(chunk)

        well_data = self._filter_data(well_id, cur_chunk)
        X_filtered_np: np.ndarray = well_data[self.all_features]
        y_filtered_np: np.ndarray = well_data["phi"]

        # Convert from structured array to simple array
        # This conversion from array->list->array may be inefficient...
        X_filtered_np = np.array(X_filtered_np.tolist())
        y_filtered_np = np.array(y_filtered_np.tolist())

        return X_filtered_np, y_filtered_np

    def _get_target_chunk(self, chunk: int = -1) -> np.ndarray:
        """
        Returns the target chunk from self.cur_h5_dset.
        If the chunk is not passed, returns the whole data
        """
        if chunk < 0:
            cur_chunk = self.cur_h5_dset[:]
        else:
            # Seeks the chunk to be read
            # TODO: Improve way of finding the correct chunk.
            cur_slice_it = self.cur_h5_dset.iter_chunks()
            cur_slice = cur_slice_it.__next__()
            for _ in range(chunk):
                cur_slice = cur_slice_it.__next__()

            # Load the hypercube chunk
            cur_chunk = self.cur_h5_dset[cur_slice]
        return cur_chunk

    def _filter_data(self, well_id: int, cur_chunk):
        """
        Return all data not associated with well_id.
        If well_id == -1, all of the data are returned.
        """
        if well_id != -1:
            well_data = cur_chunk[cur_chunk["well_id"] != well_id]
        else:
            well_data = cur_chunk
        return well_data


# Only values within the current iteration are found
def get_count_if_field_is(h5_dset, it, wells, field, vals_to_cmp):
    metrics = np.zeros(len(vals_to_cmp))

    # Get all metrics for each chunk
    for chunk_slice in h5_dset.iter_chunks():
        has_data = False
        for well_coords in wells:
            well_coords_slice = (slice(well_coords[0] - it,
                                       well_coords[0] + it),
                                 slice(well_coords[1] - it,
                                       well_coords[1] + it))
            # Get metrics for
            if slice_overlaps_xy(chunk_slice, well_coords_slice):
                has_data = True
                continue

        if has_data:
            chunk_np = h5_dset[chunk_slice]
            for i in range(len(metrics)):
                metrics[i] += (chunk_np[field] == vals_to_cmp[i]).sum()

    return [int(m) for m in metrics]


def main():
    parser = argparse.ArgumentParser(description="usage: H5_FILE [OPTS]")

    parser.add_argument('filename')

    parser.add_argument(
        "-i",
        dest="it_info",
        action="store",
        default=None,
        help="Show h5 info. Need to give the max it expected. Data outside "
        "this ring will not be considered.",
    )

    parser.add_argument(
        "-c",
        dest="it_clear",
        action="store",
        default=None,
        help=
        "Clear all data which is not inside the inputted ring (inclusive).",
    )

    parser.add_argument(
        "-p",
        dest="it_prop",
        action="store",
        default=None,
        help="Propagate data until inputted ring.",
    )

    args = parser.parse_args()

    # Get porosity H5 file
    print(args.filename)
    h5_file = h5py.File(args.filename, 'r+')
    if not h5_file:
        print(f"Could not open {args.filename}")
        return
    h5_dset = h5_file['p']

    if args.it_info:
        it_info = int(args.it_info) + 1
        # get shape
        print(f"Hypercube shape: {h5_dset.shape}")

        # get chunking
        chunks = [ceil(a / b) for (a, b) in zip(h5_dset.shape, h5_dset.chunks)]
        print(f"Hypercube chunks: {chunks}")
        print(f"Hypercube chunk shape: {h5_dset.chunks}")

        # get wells
        ring0 = h5_dset[h5_dset['real'] == common.RealValues.real][['x', 'y']]
        #wells = []
        #for x in np.unique(ring0['x']):
        #    wells.append((x, ring0[ring0['x'] == x][0]['y']))
        wells = set([(xx, yy) for xx, yy in ring0.tolist()])
        print("Wells:")
        print(list(wells))

        # get types of points counts
        metrics = get_count_if_field_is(h5_dset, it_info, wells, 'real', [
            common.RealValues.real, common.RealValues.propagated,
            # common.RealValues.canal, common.RealValues.canal_expanded,
            # common.RealValues.expanded
        ])

        # get types of points counts
        wells_metrics = get_count_if_field_is(h5_dset, it_info, wells,
                                              'well_id', range(10))

        print("Points per well:")
        print(wells_metrics)

        total_points = prod(h5_dset.shape)
        print(f"Total points: {total_points}")
        print(f"\tReal: {metrics[0]} {100*(metrics[0]/total_points):.2f}%")
        print(f"\tPropagated: {metrics[1]} "
              f"{100*(metrics[1]/total_points):.2f}%")
        # print(f"\tCanal: {metrics[2]} {100*(metrics[2]/total_points):.2f}%")
        # print(f"\tExpanded: {metrics[2]} "
        #       f"{100*(metrics[2]/total_points):.2f}%")
        # print(f"\tCanal expanded: {metrics[4]} "
        #       f"{100*(metrics[4]/total_points):.2f}%")
        empty_points = total_points - sum(metrics)
        print(
            f"\tEmpty: {empty_points} {100*(empty_points/total_points):.2f}%")

        
        max_ring = 0
        
        for chunk_slice in h5_dset.iter_chunks():

            chunk_np = h5_dset[chunk_slice]
            max_ring = max(max_ring, np.max(chunk_np['ring']))

        print(f"MaxRing {max_ring}")


        # # get max iteration
        # print("Points per iteration:")
        # metrics = get_count_if_field_is(h5_dset, it_info, wells, 'ring',
        #                                 list(range(it_info)))
        # for i, count in enumerate(metrics):
        #     print(f"\tit{i}: {count}")

    if args.it_clear:
        for chunk_slice in h5_dset.iter_chunks():
            np_dset = h5_dset[chunk_slice]

            # print(f"updating {chunk_slice}")

            def _update(np_dset, cond, default):
                return np.where(cond, default, np_dset)

            cond = np_dset['ring'] > int(args.it_clear)
            # print(f'to update: {cond.sum()}')
            np_dset['phi'] = _update(np_dset['phi'], cond, 0)
            np_dset['real'] = _update(np_dset['real'], cond,
                                      common.RealValues.empty)
            np_dset['ring'] = _update(np_dset['ring'], cond, -1)
            np_dset['well_id'] = _update(np_dset['well_id'], cond, -1)
            h5_dset[chunk_slice] = np_dset


if __name__ == '__main__':
    main()
