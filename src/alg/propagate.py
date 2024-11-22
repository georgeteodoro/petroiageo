from dataclasses import dataclass
from h5py import Dataset
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error
import h5py
from math import prod, sqrt
import argparse
import ast
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import common
import mpi_module
import config_parser
from feature_data.FeatureDatasetBase import FeatureDatasetBase
from TrialDataBase import TrialDataBase
from TrialDataNumpy import TrialDataNumpy
from feature_data.FeatureDatasetMMapCache import FeatureDatasetMMapCache


@dataclass
class ModelEval:
    """
    full_rmse: RMSE of data
    full_mae: MAE of data
    wells_rmse: Dict of well idx as key and its RMSE as value
    wells_mae: Dict of well idx as key and its MAE as value
    """
    full_rmse: float
    full_mae: float
    wells_rmse: dict
    wells_mae: dict


def propagate(porosity_data_h5: Dataset, trial_data: TrialDataBase,
              all_features: FeatureDatasetBase, best_features: list, it: int,
              config: config_parser.Config) -> int:
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

    rank = config.get_param('mpi_rank')

    print(f"[propagation][it{it}] Propagating with: {best_features}")

    # Only train coords should be propagated, test wells shouldn't
    wells_coords = config.train_wells_coords
    train_wells_ids = config.train_wells_ids

    # First iteration is 1, but first ring is 0. However, the following ring
    # should be propagated, resulting in 'ring = it - 1 + 1'.
    ring = it

    # Prepare the model and evaluate its performance metrics
    model = _train_model(trial_data)

    pov_canal_path = config.get_param('pov_canal_path')

    if pov_canal_path == None:
        model_eval = _test_model(porosity_data_h5, all_features, best_features,
                                 config, model)
    else:
        model_eval = _pov_test_model(pov_canal_path, all_features,
                                     best_features, config, model, it)
    print(f"[propagation][it{it}] Test errors: RMSE: {model_eval.full_rmse} " +
          f"MAE: {model_eval.full_mae}")
    print(f"[propagation][it{it}] Test wells performance: RMSE: "
          f"{model_eval.wells_rmse} " + f"MAE: {model_eval.wells_mae}")

    # Count of propagated points for checking if it was correct
    n_propagated_points = 0

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

            coords_to_update = _get_coords_to_propagate(
                ring, cur_chunk_np, w_x, w_y, config)

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


def _get_coords_to_propagate(target_ring: int, data: np.ndarray, w_x: int,
                             w_y: int, config) -> np.ndarray:
    """
    Get data's point's coordinates suitable for propagation on ring target_ring 
    relative to the well in w_x, w_y

    target_ring: Integer representing the ring starting from the well position
    where to look for suitable points for propagation
    data: The data containing the points
    w_x: Integer with the well's x location
    w_y: Integer with the well's y location

    Return:
    3 dimensional np.ndarray with the coordinates of point's suitable for 
    propagation. The dimensions are relative to the x,y and z dimensions 
    respectively

    """
    # Calculate the coordinates of the current well-ring
    well_ring_x_left = w_x - target_ring
    well_ring_x_right = w_x + target_ring
    well_ring_y_top = w_y - target_ring
    well_ring_y_bot = w_y + target_ring

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

    r_val = common.RealValues.empty

    pov_canal_path = config.get_param('pov_canal_path')
    pov_should_prop = config.get_param('pov_should_prop')
    if pov_canal_path != None:
        r_val = common.RealValues.canal
        if not pov_should_prop:
            r_val = common.RealValues.none

    # Only empty points can be propagated
    filter_fun = lambda d: (d['real'] == r_val) \
                                 & (left_wall_cond(d) \
                                  | right_wall_cond(d) \
                                  | top_wall_cond(d) \
                                  | bot_wall_cond(d))
    coords_to_update = np.where(filter_fun(data))
    return coords_to_update


def _train_model(trial_data: TrialDataBase):
    '''
    Generate a LightGBM model for estimating porosity.
    Return the model
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

    # model.save_model('model.txt')

    return model


def _predict_data(model, all_features: FeatureDatasetBase, best_features: list,
                  coords_to_update: np.ndarray):
    '''
    Use the model to predict the porosity values based on the best_features at 
    coords_to_update.

    model: The model used to predict the porosities. The model must have a
    predict() function.
    all_features: FeatureDatasetBase that has access to all features
    best_features: List of features to select from all_features to be used
    by the model
    coords_to_update: Target coords where to get the best_features from

    Return
    The predicted porosity measures
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
        feature_coords_np = np.array(list(zip(*feature_coords)),
                                     dtype=np.int64)

        # Filter features values for current chunk coords
        to_predict_np[:, f_id] = all_features.get_feature(
            feature).filter_coords(feature_coords_np)

    # Perform porosity estimation
    estimated_phi = model.predict(to_predict_np)

    return estimated_phi


def _test_model(data: Dataset, all_features: FeatureDatasetBase,
                best_features: list, config: config_parser.Config,
                model) -> ModelEval:
    """
    Apply the model on the test data inside data.
    Return a ModelEval instance
    """
    #Create and prepare the test data
    test_data = TrialDataNumpy(config.test_wells_ids,
                               data,
                               config,
                               should_consider_sampling=False)
    # As test data dont propagate, the test data is always at the prep_it=1
    test_data.prepare_porosity(1)
    for (feature, disp) in best_features[:-1]:
        test_data.commit_feature(all_features.get_feature(feature), disp)
    (feature, disp) = best_features[-1]
    test_data.update_feature(all_features.get_feature(feature), disp)

    return _eval_model(model, test_data, config.test_wells_ids)


def _eval_model(model, test_data: TrialDataBase,
                test_wells_ids: list) -> ModelEval:
    """
    Evaluate the model on the test_data based on the test_wells_ids.
    The model must have a predict(X_test) method.
    Return an instance of ModelEval
    """
    rmse_per_well = dict()
    mae_per_well = dict()
    n_instances = 0
    sse = 0
    sum_abs_errors = 0
    for well_id in test_wells_ids:
        X_test, Y_test = test_data.get_val_values(well_id)
        n_instances += len(Y_test)
        # There are no data from this well on test data for some reason
        msg = "[propagate][_eval_model] There are no test data"
        msg += f" for well id {well_id}"
        assert len(X_test) > 0, msg

        pred = model.predict(X_test)

        #RMSE
        curr_sse = np.sum((Y_test - pred)**2)

        sse += curr_sse

        well_mse = curr_sse / len(Y_test)
        rmse_per_well[well_id] = float(np.sqrt(well_mse))

        #MAE
        curr_abs = np.abs(Y_test - pred)
        sum_abs_errors += np.sum(curr_abs)

        well_mae = mean_absolute_error(Y_test, pred)
        mae_per_well[well_id] = float(well_mae)

    # Sqrt of means is different from mean of sqrts. The former is what we want
    rmse = np.sqrt(sse / n_instances)
    mae = sum_abs_errors / n_instances

    model_eval = ModelEval(rmse, mae, rmse_per_well, mae_per_well)
    return model_eval


def _get_ring_values(canal_dset: Dataset, ring, well):
    '''
    How to avoid adding duplicate points: 
    for instance for a 4x4 area: 
    top(t), bottom(b), left(l), right(r)
    l t t t
    l     r
    l     r
    b b b r
    '''

    # Retrieve ring values
    x, y = well
    top = canal_dset[x - ring + 1:x + ring + 1, y - ring]
    bottom = canal_dset[x - ring:x + ring, y + ring + 1]
    left = canal_dset[x - ring, y - ring:y + ring]
    right = canal_dset[x + ring + 1, y - ring + 1:y + ring + 1]

    # Reshape to a 1D array
    top.reshape(prod(top.shape))
    bottom.reshape(prod(bottom.shape))
    left.reshape(prod(left.shape))
    right.reshape(prod(right.shape))

    ring_points = np.concatenate([top, bottom, left, right])

    # Filter only canal points
    ring_points = ring_points[ring_points['real'] == common.RealValues.canal]

    return ring_points


def _pov_test_model(pov_canal_path: str, all_features: FeatureDatasetBase,
                    best_features: list, config: config_parser.Config, model,
                    ring) -> ModelEval:

    canal_h5 = h5py.File(pov_canal_path)
    canal_dset = canal_h5['p']

    # Get test data for current ring and all wells
    test_data = []
    for well in config.train_wells_coords:
        test_data.append(_get_ring_values(canal_dset, ring, well))
    test_data = np.concatenate(test_data)

    # Retrieve Y
    Y_test = test_data['phi']

    # Generate the input features
    X_test = []
    X_test_coords = test_data[['x', 'y', 'z']]
    for feature, disp in best_features:
        for i in range(len(X_test_coords)):
            x, y, z = X_test_coords[i]
            X_test_coords[i] = (x + disp[0], y + disp[1], z + disp[2])

        f_values = all_features.get_feature(feature).filter_coords(
            X_test_coords)

        X_test.append(f_values)

    # Merge X_test and reshape it to the features shape
    X_test = np.concatenate(X_test).reshape((len(X_test[0]), len(X_test)))

    pred = model.predict(X_test)

    #RMSE
    sse = np.sum((Y_test - pred)**2)
    mse = sse / len(Y_test)
    rmse = float(np.sqrt(mse))

    #MAE
    mae = mean_absolute_error(Y_test, pred)

    return ModelEval(rmse, mae, None, None)


def main(args_str=None):
    parser = argparse.ArgumentParser(description="Ferramenta de propagação")
    parser.add_argument(
        '--config',
        dest='config_file',
        action='store',
        required=True,
        type=str,
        help="The yaml config file path to be read",
    )
    parser.add_argument(
        '--features',
        dest='features_sets',
        action='store',
        required=True,
        type=str,
        help="Path of the features sets to execute. For each line, a "
        "propagation iteration will be ran, except if asked to run fewer "
        "iterations.",
    )
    parser.add_argument(
        '--it',
        dest='start_it',
        action='store',
        required=False,
        type=int,
        default=0,
        help="Initial iteration to begin (inclusive) (default=0).",
    )
    parser.add_argument(
        '--nits',
        dest='num_its',
        action='store',
        required=False,
        type=int,
        default=0,
        help="Number of iterations to run (default=0, i.e., all).",
    )
    parser.add_argument(
        '--gab',
        dest='gab',
        action='store',
        default=None,
        required=False,
        type=str,
        help="Gabarito.",
    )
    parser.add_argument(
        '--pov',
        dest='pov_canal_path',
        action='store',
        default=None,
        required=False,
        type=str,
        help="Configures propagation and validation for POV. Input path "
        "is from canal data. Canal data should be an h5 file. This file "
        "is opened in read-only mode. POV must start on it=1.",
    )
    parser.add_argument(
        '--pov-no-prop',
        dest='pov_should_prop',
        action='store_false',
        default=True,
        required=False,
        help="Only usable with --pov. When propagating, estimated porosity "
        "will only be used for test. Afterward, original canal points are "
        "reloaded int the H5 porosity structure.",
    )
    parser.add_argument(
        '--jv',
        dest='just_validate',
        action='store_true',
        default=False,
        required=False,
        help="Just validate the propagated results. (don't propagate).",
    )

    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split(' '))

    config = config_parser.YAMLConfig(args.config_file)
    config.add_param('pov_canal_path', args.pov_canal_path)
    config.add_param('n_training_chunks', 1)
    config.add_param('pov_should_prop', args.pov_should_prop)
    mpi_module.initialize(config)

    # Parse propagations to be ran
    features_sets = []
    with open(args.features_sets) as f_sets:
        for f_set in f_sets:
            features_sets.append(ast.literal_eval(f_set))

    # Find largest number of features required
    n_features = 0
    for f_set in features_sets:
        n_features = max(len(f_set), n_features)
    config.alg['max_num_features'] = n_features

    # Set initial and end 'it' to be ran
    it = args.start_it
    features_sets = features_sets[it:]
    if args.num_its > 0:
        features_sets = features_sets[:args.num_its]

    print(f'[propagation][it{it}] preparing trial data')

    # Create TrialData
    porosity_h5_f = h5py.File(config.starting_porosity_cube_path, 'r+')
    porosity_h5_dset = porosity_h5_f[common.POROSITY_DSET_NAME]
    trial_data = TrialDataNumpy(config.train_wells_ids, porosity_h5_dset,
                                config)
    trial_data.prepare_porosity(it)

    all_features = FeatureDatasetMMapCache(config)

    print(f'[propagation][it{it}] starting propagations...')

    if args.just_validate:
        trial_data.prepare_porosity(it + args.num_its)
    else:
        # Run propagations
        for f_set in features_sets:
            # Add features. Last feature needs to be added without a commit
            # since commit increments the features' counter of trial_data.
            # E.g., if 2 features were used on a trial_data with up to 4
            # features, the feature counter would end on 3 if only
            # commit_feature was used.
            print(f'[propagation][it{it}] committing features')
            for (feature, disp) in f_set[:-1]:
                trial_data.commit_feature(all_features.get_feature(feature),
                                          disp)
            (feature, disp) = f_set[-1]
            trial_data.update_feature(all_features.get_feature(feature), disp)

            print(f'[propagation][it{it}] performing propagation')
            n_prop_points = propagate(porosity_h5_dset, trial_data,
                                      all_features, f_set, it, config)
            print(f"[propagation][it{it}] propagated {n_prop_points} points")
            it += 1
            trial_data.prepare_porosity(it)

    # Validate propagated data
    if args.gab != None:
        gab = pd.read_csv(args.gab)

        print("Filtering propagated points")
        prop_points = porosity_h5_dset[porosity_h5_dset['real'] ==
                                       common.RealValues.propagated]

        print("Preparing gab")
        gab = gab[['x', 'y', 'z', 'phi']].to_numpy()

        # Index gab
        hier_gab = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for x, y, z, phi in tqdm(gab, total=len(gab)):
            hier_gab[x][y][z] = phi

        print("Calculating diffs")
        diffs = np.empty(len(prop_points))
        for i, (x, y, z,
                phi) in tqdm(enumerate(prop_points[['x', 'y', 'z', 'phi']]),
                             total=len(prop_points)):
            diffs[i] = phi - hier_gab[x][y][z]

        # # Slow
        # diffs = np.empty(len(gab))
        # gab = gab[['x', 'y', 'z', 'phi']].to_numpy()

        # for i, (x, y, z, Y) in tqdm(enumerate(gab), total=len(gab)):
        #     diffs[i] = porosity_h5_dset[(int(x), int(y), int(z))]['phi'] - Y

        sae = 0
        sse = 0
        with open('diffs.txt', 'w') as d_file:
            for i in range(len(diffs)):
                sae += abs(diffs[i])
                sse += diffs[i]**2
                d_file.write(f'{diffs[i]}\n')

        print(f"mae: {sae/len(diffs)}")
        print(f"rmse: {sqrt(sse/len(diffs))}")

    porosity_h5_f.close()


def wrapper(p_hp5, c):
    x, y, z, Y = c
    return Y - p_hp5[(int(x), int(y), int(z))]['phi']


if __name__ == '__main__':
    main()
