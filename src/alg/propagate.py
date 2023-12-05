import numpy as np

import common


def _predict_data(model, features_dict, best_features, coords_to_update,
                  config):
    '''
    Generate the porosity values of 'coords_to_update'.
    '''

    window_size = config.alg["window"]
    disp_cube_shape = (
        window_size * 2 + 1,
        window_size * 2 + 1,
        window_size * 2 + 1,
    )

    # Create an ndarray for keeping all features
    n_points_to_propagate = len(coords_to_update[0])
    predict_features_type = [(f"f{f}", np.float64)
                             for f in range(len(best_features_set))]
    to_predict_np = np.empty(n_points_to_propagate,
                             dtype=predict_features_type)

    # Generator function to filter features with a given coords list
    # A generator is used to avoid iterating point by point on a given feature.
    # For H5, each individual access has high overhead. But by giving a
    # generator to it, all operations are performed with reduced overhead.
    def _gen_list_features(feature_dset, coords_3d_np):
        for coord in coords_3d_np:
            yield feature_dset[coord]

    # Fill the values of each feature
    for (feature, disp) in best_features_set:
        # Apply the displacement one coord at a time
        feature_coords = coords_to_update.copy()
        for d_id, coord_s in enumerate(["x", "y", "z"]):
            feature_coords[coord_s] = (feature_coords[coord_s] + disp[d_id] +
                                       ((disp_cube_shape[d_id] - 1) / 2))

        # Zip the coords, from a tuple of 3 arrays, one for each coord,
        # to an array of (x,y,z) tuples.
        feature_coords = np.array(zip(*feature_coords))

        # Filter features values for current chunk coords
        feature_values = _gen_list_features(features_dict_h5[feature],
                                            feature_coords)


        ##############:
        # move _gen_list_features to FeatureBase
        # add a new method to return the features' values given feature_coords


def propagate(porosity_data_h5, features_dict, best_features, it, config):
    '''
    Propagates the wavefront a single ring. Initial data have no 
    'expanded' data.
    Currently, porosity_data has no encapsulation, thus it is operated upon
    directly. If encapsulating class is created, it must begin here.
    '''

    # Only train coords should be propagated, test wells shouldn't
    wells_coords = config.train_wells_coords
    train_wells_ids = config.train_wells_ids
    test_wells_ids = config.alg["test_only_wells"]

    # First iteration is 1, but first ring is 0. However, the following ring
    # should be propagated, resulting in 'ring = it - 1 + 1'.
    ring = it

    # Prepare the model and evaluate its performance metrics
    # model = _train_model()
    # rmse, mae = _eval_model(model)
    # print(f"[propagation][it{it}] RMSE: {rmse}, MAE: {mae}")

    # Propagate on all chunks from porosity_data
    for chunk_n, cur_slice in enumerate(porosity_data_h5.iter_chunks()):

        # Get the list of wells with points to update within the current chunk
        wells_to_update = common.has_points_within_chunk(wells_coords,
                                                         ring,
                                                         cur_slice,
                                                         return_list=True)

        # If there are no points to propagate, skip this chunk
        if len(wells_to_update) == 0:
            continue

        # Load porosity data of the current chunk since there are points
        # to be propagated
        cur_chunk_np = porosity_data_h5[cur_slice]

        # Propagate the points of each well
        for w_x, w_y in wells_to_update:

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

            # Only empty points can be propagated
            filter_fun = lambda d: (d['real'] == common.RealValues.empty) \
                                 & (left_wall_cond(d) \
                                  | right_wall_cond(d) \
                                  | top_wall_cond(d) \
                                  | bot_wall_cond(d))
            coords_to_update = np.where(filter_fun(cur_chunk_np))

            print(coords_to_update)
            0 / 0

            # Update the filtered values on the tmp nparray
            cur_chunk_np['real'][coords_to_update] = common.RealValues.propagated
            cur_chunk_np['ring'][coords_to_update] = ring
            cur_chunk_np['well_id'][coords_to_update] = train_wells_ids[
                wells_coords.index((w_x, w_y))]
            cur_chunk_np['phi'][coords_to_update] = _predict_data(
                model, features_dict, best_features, coords_to_update, config)

            # Commit update to the h5 file
            porosity_data_h5["real", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["real"]
            porosity_data_h5["ring", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["ring"]
            porosity_data_h5["well_id", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["well_id"]
            porosity_data_h5["phi", cur_slice[0], cur_slice[1],
                             cur_slice[2]] = cur_chunk_np["phi"]
