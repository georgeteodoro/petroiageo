from dataclasses import dataclass
from h5py import Dataset
import lightgbm as lgb
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

import common
from config_parser import Config
from TrialDataBase import TrialDataBase
from TrialDataNumpy import TrialDataNumpy
from data_filter import WellsDataFilter


@dataclass
class ModelEval:
    full_rmse: float
    full_mae: float
    wells_rmse: dict
    wells_mae: dict


def _train_model(trial_data: TrialDataBase):
    '''
    Generate a LightGBM model for estimating porosity.
    '''
    model = None
    chunk_id = 0  # No incremental learning yet, so just return the first chunk

    # Generate a training dataset for all data
    X_train_np, y_train_np = trial_data.get_train_values(well_id=-1,
                                                         chunk_id=chunk_id)
    lgb_train_dataset = lgb.Dataset(X_train_np, y_train_np)

    # Perform training
    model = lgb.train(
        common.training_params,
        lgb_train_dataset,
        init_model=model,
        num_boost_round=100,
        keep_training_booster=True,
    )

    return model


def _eval_model(model, test_data: TrialDataBase,
                test_wells_ids: list) -> ModelEval:
    """
    Evaluate the model on the test_data based on the test_wells_ids.
    The model must have a predict(X_test) method.
    Return the an instance of ModelEval
    """
    mse = list()
    mae = list()
    rmse_per_well = dict()
    mae_per_well = dict()
    for well_id in test_wells_ids:
        X_test, Y_test = test_data.get_val_values(well_id)

        # There are no data from this well on test data for some reason
        msg = "[propagate][_eval_model] There are no test data"
        msg += f" for well id {well_id}"
        assert len(X_test) > 0, msg

        pred = model.predict(X_test)

        well_mse = np.mean((pred - Y_test)**2)
        rmse_per_well[well_id] = np.sqrt(well_mse)
        mse.append(well_mse)

        well_mae = mean_absolute_error(Y_test, pred)
        mae_per_well[well_id] = well_mae
        mae.append(well_mae)

    # Sqrt of means is different from mean of sqrts. The former is correct
    rmse = np.sqrt(np.mean(mse))

    model_eval = ModelEval(rmse, np.mean(mae), rmse_per_well, mae_per_well)
    return model_eval


def _predict_data(model, all_features, best_features, coords_to_update):
    '''
    Generate the porosity values of 'coords_to_update'.
    '''

    # Create an ndarray for keeping all features
    n_points_to_propagate = len(coords_to_update[0])
    to_predict_np = np.empty((n_points_to_propagate, len(best_features)),
                             dtype=np.float64)

    # Fill the values of each feature
    for f_id, (feature, disp) in enumerate(best_features):
        # Apply the displacement one coord at a time
        # feature_coords = coords_to_update.copy()
        feature_coords = []
        for d_id, coords in enumerate(coords_to_update):
            feature_coords.append(coords + disp[d_id])

        # Zip the coords, from a tuple of 3 arrays, one for each coord,
        # to an array of (x,y,z) tuples.
        feature_coords_np = np.array(list(zip(*feature_coords)), dtype=np.int64)

        # Filter features values for current chunk coords
        to_predict_np[:, f_id] = all_features.get_feature(
            feature).filter_coords(feature_coords_np)

    # Perform porosity estimation
    estimated_phi = model.predict(to_predict_np)

    return estimated_phi


def propagate(porosity_data_h5: Dataset, trial_data: TrialDataBase,
              all_features, best_features: list, it: int,
              config: Config) -> int:
    '''
    Propagates the wavefront a single ring. Initial data have no 
    'expanded' data.
    Currently, porosity_data has no encapsulation, thus it is operated upon
    directly. If encapsulating class is created, it must begin here.
    The 'trial_data' input is from the feature selection process, and thus have 
    all the features and porosity already set up. It is then used to generate
    the model for porosity estimation.
    Returns the number of propagated points.
    '''

    window_size = config.alg["window"]
    hypercube_shape = porosity_data_h5.shape

    rank = config.get_param('mpi_rank')

    # Only train coords should be propagated, test wells shouldn't
    wells_coords = config.train_wells_coords
    train_wells_ids = config.train_wells_ids

    # First iteration is 1, but first ring is 0. However, the following ring
    # should be propagated, resulting in 'ring = it - 1 + 1'.
    ring = it

    # Prepare the model and evaluate its performance metrics
    model = _train_model(trial_data)

    #Create and prepare the test data
    test_data = TrialDataNumpy(config.test_wells_ids,
                               porosity_data_h5,
                               config,
                               should_consider_sampling=False)
    # As test data dont propagate, the test data is always at the prep_it=1
    test_data.prepare_porosity(1)
    for (feature, disp) in best_features:
        test_data.commit_feature(all_features.get_feature(feature), disp)

    model_eval = _eval_model(model, test_data, config.test_wells_ids)
    print(f"[propagation][it{it}] Test errors: RMSE: {model_eval.full_rmse} " +
          f"MAE: {model_eval.full_mae}")
    print(
        f"[propagation][it{it}] Test wells performance: RMSE: {model_eval.wells_rmse} "
        + f" MAE: {model_eval.wells_mae}")

    # Count of propagated points for checking if it was correct
    n_propagated_points = 0

    # # File used for debugging, should be empty at the beginning of the iteration
    # import os
    # cur_chunk_np_filename = f'cur_chunk_np-w{rank}.txt'
    # if os.path.exists(cur_chunk_np_filename):
    #     os.remove(cur_chunk_np_filename)

    # Propagate on all chunks from porosity_data
    for chunk_n, cur_slice in enumerate(porosity_data_h5.iter_chunks()):
        # print(f"[propagation0][it{it}][worker{rank}] "
        #       f"Chunk {chunk_n}:{cur_slice}")

        # Get the list of wells with points to update within the current chunk
        wells_to_update = common.has_points_within_chunk(wells_coords,
                                                         ring,
                                                         cur_slice,
                                                         return_list=True)

        # print(f"[propagation0][it{it}][worker{rank}] len(wells_to_update): "
        #       f"{len(wells_to_update)}")

        # If there are no points to propagate, skip this chunk
        if len(wells_to_update) == 0:
            # print(f"[propagation0][it{it}][worker{rank}] Chunk skipped.")
            continue

        # Load porosity data of the current chunk since there are points
        # to be propagated
        cur_chunk_np = porosity_data_h5[cur_slice]
        # print(f"[propagation1][it{it}][worker{rank}][chunk{chunk_n}] "
        #       f"cur_chunk_np hash: {hash(cur_chunk_np.data.tobytes())}")

        # Propagate the points of each well
        for w_x, w_y in wells_to_update:
            # print(f"[propagation1][it{it}][worker{rank}][chunk{chunk_n}] "
            #       f"well: {(w_x, w_y)}")

            # Calculate the coordinates of the current well-ring
            well_ring_x_left = w_x - ring
            well_ring_x_right = w_x + ring
            well_ring_y_top = w_y - ring
            well_ring_y_bot = w_y + ring

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

            # Condition to avoid padding region since the displacement
            # may result in out of bounds access.
            not_on_padding = (lambda d: (d['z'] >= window_size) &
                              (d['z'] < hypercube_shape[2] - window_size) &
                              (d['y'] >= window_size) &
                              (d['y'] < hypercube_shape[1] - window_size) &
                              (d['x'] >= window_size) &
                              (d['x'] < hypercube_shape[0] - window_size))

            # Only empty points can be propagated
            filter_fun = lambda d: (d['real'] == common.RealValues.empty) \
                                 & (left_wall_cond(d) \
                                  | right_wall_cond(d) \
                                  | top_wall_cond(d) \
                                  | bot_wall_cond(d)) \
                                 & not_on_padding(d)
            coords_to_update = np.where(filter_fun(cur_chunk_np))

            n_propagated_points += len(coords_to_update[0])

            # print(f"[propagation1][it{it}][worker{rank}][chunk{chunk_n}] "
            #       f"n_propagated_points: {n_propagated_points}")

            # Update the filtered values on the tmp nparray
            cur_chunk_np['real'][
                coords_to_update] = common.RealValues.propagated
            cur_chunk_np['ring'][coords_to_update] = ring
            cur_chunk_np['well_id'][coords_to_update] = train_wells_ids[
                wells_coords.index((w_x, w_y))]
            cur_chunk_np['phi'][coords_to_update] = _predict_data(
                model, all_features, best_features, coords_to_update)

            # Commit update to the h5 file
            porosity_data_h5["real", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["real"]
            porosity_data_h5["ring", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["ring"]
            porosity_data_h5["well_id", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["well_id"]
            porosity_data_h5["phi", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["phi"]

    return n_propagated_points
