import h5py
from typing import Dict
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from time import time
from typing import Tuple, Dict
import numpy as np

from inverted_learning_interface import AbstractApplyAlg
import config_parser
import common
from h5py import File
import profiling
import hdf5_util
import petro5_hdf5
from data_filter import PredTrainDataFilter

# TODO: send these values to self._config
params = {
    "max_bin": 128,
    "max_depth": 10,
    "learning_rate": 0.1,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 20,
    "verbose": -1,
    "min_data": 10,
    "boost_from_average": True,
    "bagging_freq": 1,
    "random_state": 0,
}


class H5ApplyAlg(AbstractApplyAlg):
    """
    Apply algorithm for HDF5 files and MPI.
    Although MPI support is there, it is possible to run it without 'mpirun'.

    Update of porosity values is done in-place.
    """

    def __init__(self, config: config_parser.Config):
        self._config = config

        # Compatibility flags:
        super().__init__()
        self._using_h5 = True

    def perform_prediction(
        self,
        best_features_set: set,
        features_dict_h5: Dict[str, h5py.Dataset],
        porosity_data_h5: h5py.Dataset,
        it: int,
    ):
        # Retrieve self._config parameters
        should_update = self._config.get_param("mpi_should_update_local")

        # Only one process per node is required to update
        if should_update:
            window_size = self._config.alg["window"]
            # Calculate remaining variables
            displacement_cube_shape = (
                window_size * 2 + 1,
                window_size * 2 + 1,
                window_size * 2 + 1,
            )

            # Remove coordinates from features set
            best_features_set.remove("x")
            best_features_set.remove("y")
            best_features_set.remove("z")

            cur_h5, test_h5, train_list, test_list = self._get_train_test_dset_list(
                best_features_set, features_dict_h5, porosity_data_h5, it,
                displacement_cube_shape)

            regressor = self._train_regressor(it, train_list)
            cur_h5.close()

            final_rmse, final_mae = self._evaluate_regressor(
                test_list, regressor)
            test_h5.close()

            msg = f"[manager][it{it}][test-error] RMSE: {final_rmse}"
            msg += f" MAE: {final_mae}"
            print(msg)

            t3 = time()

            # Perform prediction of expanded points

            # Define a function to filter only the expanded points
            is_to_pred_point_f = lambda d: (
                (d["real"] == common.RealValues.canal_expanded)
                | (d["real"] == common.RealValues.expanded))
            p_sum = 0
            for chunk_n, cur_slice in enumerate(
                    porosity_data_h5.iter_chunks()):
                t4 = time()

                cur_chunk_np = porosity_data_h5[cur_slice]

                # Check if there is any point on the current chunk to be updated
                # TODO:
                # Extract check for overlapping rectangles in
                # expand.gen_expanded_points, parameterize by
                # (it, window_size, slice) and use it here
                is_chunk_pred_points_list = is_to_pred_point_f(cur_chunk_np)
                if not is_chunk_pred_points_list.any():
                    continue

                # Filter points to predict
                expanded_points_np: np.ndarray = cur_chunk_np[
                    is_chunk_pred_points_list]

                to_propagate_count = len(expanded_points_np)
                p_sum = p_sum + to_propagate_count

                to_predict_np = self._get_data_to_predict(
                    best_features_set,
                    features_dict_h5,
                    displacement_cube_shape,
                    expanded_points_np,
                    to_propagate_count,
                )

                t5 = time()
                profiling.prof_predict_pred_insert_time(
                    it, chunk_n, t5 - t4, self._config)

                # Perform prediction of expanded points
                new_phi_np = regressor.predict(to_predict_np)

                t6 = time()
                profiling.prof_predict_pred_run_time(it, chunk_n, t6 - t5,
                                                     self._config)

                # Update porosity values on cur_chunk_np. This serves 2
                # purposes: (i) converts the new_phi_np from a list to a
                # 3d array (cur_chunk_np) on the proper coordinates and,
                # (ii) allows batched write of data onto porosity_data_h5
                updated_coords = np.where(is_chunk_pred_points_list)
                cur_chunk_np["phi"][updated_coords] = new_phi_np
                cur_chunk_np["real"][
                    updated_coords] = common.RealValues.propagated

                # Only update 'phi' and 'real' values of expanded points
                porosity_data_h5["phi", cur_slice[0], cur_slice[1],
                                 cur_slice[2]] = cur_chunk_np["phi"]
                porosity_data_h5["real", cur_slice[0], cur_slice[1],
                                 cur_slice[2]] = cur_chunk_np["real"]

                t7 = time()
                profiling.prof_predict_pred_update_time(
                    it, chunk_n, t7 - t6, self._config)
                profiling.prof_predict_pred_time(it, chunk_n, t7 - t4,
                                                 self._config)

            t8 = time()
            profiling.prof_predict_pred_times(it, t8 - t3, self._config)
        else:
            my_rank = self._config.get_param("mpi_rank")
            print(f"[main][{it}][R{my_rank}] waiting points propagation")

        comm = self._config.get_param("mpi_global_comm")
        comm.Barrier()

    def _evaluate_regressor(self, test_list: hdf5_util.HDFMultiColList,
                            regressor: lgb.Booster) -> Tuple[float, float]:
        """
        Evaluate the regressor on test data. Returns the RMSE and MAE error
        """
        final_rmse = float("inf")
        final_mae = float("inf")
        if test_list is not None:
            squared_errors = np.array([])
            mae_list = list()
            for c in range(test_list.n_chunks):
                # Generate a test dataset for all data on chunk c
                X_test_np, y_test_np = test_list.get_data_not_in_well(c)
                pred = regressor.predict(X_test_np)
                squared_errors = np.concatenate(
                    [squared_errors, (pred - y_test_np)**2])
                mae = mean_absolute_error(y_test_np, pred)
                mae_list.append(mae)

            final_rmse = np.sqrt(np.mean(squared_errors))
            final_mae = np.mean(mae_list)

        return final_rmse, final_mae

    def _get_data_to_predict(
        self,
        best_features_set: set,
        features_dict_h5: Dict[str, h5py.Dataset],
        displacement_cube_shape: tuple,
        expanded_points_np: np.ndarray,
        to_propagate_count: int,
    ) -> np.ndarray:
        # Get coordinates of points to predict
        coords_3d_np = expanded_points_np[["x", "y", "z"]]

        # Create new ndarray for keeping all features values
        # of the current chunk
        predict_features_type = [(f"f{f}", np.float64)
                                 for f in range(len(best_features_set))]
        to_predict_np = np.empty(to_propagate_count,
                                 dtype=predict_features_type)

        # Function to filter features with a given coords list
        def _gen_list_features(feature_dset, coords_3d_np):
            for coord in coords_3d_np:
                yield feature_dset[tuple(coord)]

                # Fill features values

        for f_idx, feature in enumerate(best_features_set):
            # Apply the displacement
            cur_coords_3d_np = coords_3d_np.copy()
            for d_id, coord_s in enumerate(["x", "y", "z"]):
                cur_coords_3d_np[coord_s] = (
                    cur_coords_3d_np[coord_s] + feature[d_id + 1] +
                    ((displacement_cube_shape[d_id] - 1) / 2))

            # Filter features values for current chunk coords
            feature_values = _gen_list_features(features_dict_h5[feature[0]],
                                                cur_coords_3d_np.flat)

            to_predict_np[f"f{f_idx}"] = np.fromiter(feature_values,
                                                     np.float64)

        # Convert to_predict_np from a ndarray to a regular 2d array
        to_predict_np = np.array(to_predict_np.tolist())
        return to_predict_np

    def _train_regressor(
            self, it: int,
            cur_h5_train_list: hdf5_util.HDFMultiColList) -> lgb.Booster:
        """
        Train the regressor on all data available in the last n iterations
        defined in self._config. We must ignore the points associated with 
        the testing wells.
        """
        t2 = time()

        regressor = None
        # incremental learning
        # TODO: Change this loop to go over the chunks themselves
        for c in range(cur_h5_train_list.n_chunks):
            # Generate a training dataset for all data on chunk c
            X_train_np, y_train_np = cur_h5_train_list.get_data_not_in_well(c)
            lgb_train_dataset = lgb.Dataset(X_train_np, y_train_np)

            # Perform training
            regressor = lgb.train(
                params,
                lgb_train_dataset,
                init_model=regressor,
                num_boost_round=100,
                keep_training_booster=True,
            )

        profiling.prof_predict_train_times(it, time() - t2, self._config)
        return regressor

    def _get_train_test_dset_list(
        self, best_features_set: set, features_dict_h5: Dict[str,
                                                             h5py.Dataset],
        porosity_data_h5: h5py.Dataset, it: int, displacement_cube_shape: tuple
    ) -> Tuple[File, File, hdf5_util.HDFMultiColList,
               hdf5_util.HDFMultiColList]:
        """
        Creates temporary h5 train and test structures to perform the training
        """
        rank = self._config.get_param("mpi_rank")
        t0 = time()
        data_filter = PredTrainDataFilter()

        test_only_wells = self._config.alg['test_only_wells']
        #There should be no sampling of points at this stage
        #all points from the last n iterations should be used
        #even if n == all iterations
        # We ignore the testing dset. The cur_h5_dset does not have
        # points associated with the testing wells
        cur_h5, cur_h5_dset, test_h5, test_dset_h5 = petro5_hdf5.create_tmp_dset(
            porosity_data_h5,
            data_filter,
            len(best_features_set),
            self._config,
            it,
            f'-r{rank}',
            features_only=True,
            test_wells_ids=test_only_wells,
            should_sample_max_points=False,
            generate_test_files=True)

        cur_h5_train_list = hdf5_util.HDFMultiColList(cur_h5_dset)
        cur_h5_test_list = None
        if len(test_only_wells) > 0:
            cur_h5_test_list = hdf5_util.HDFMultiColList(test_dset_h5)

        hypercube_shape = porosity_data_h5.shape

        t1 = time()
        profiling.prof_predict_create_time(it, t1 - t0, self._config)

        # Add each feature to the datasets list (TD)
        for feature in best_features_set:
            cur_h5_train_list.add_new_col()
            petro5_hdf5.insert_filtered_feature(cur_h5_dset, cur_h5_train_list,
                                                features_dict_h5, feature,
                                                hypercube_shape,
                                                displacement_cube_shape)

            if cur_h5_test_list is not None:
                cur_h5_test_list.add_new_col()
                petro5_hdf5.insert_filtered_feature(test_dset_h5,
                                                    cur_h5_test_list,
                                                    features_dict_h5, feature,
                                                    hypercube_shape,
                                                    displacement_cube_shape)

        t2 = time()
        profiling.prof_predict_insert_time(it, len(best_features_set), t2 - t1,
                                           self._config)

        return cur_h5, test_h5, cur_h5_train_list, cur_h5_test_list

    def _single_compatible(self, to_compare):
        # Check if to_compare have h5 support
        compatible = to_compare._using_h5

        return compatible
