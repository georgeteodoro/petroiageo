import pandas as pd
import numpy as np
import random
import sys
import time
from io import StringIO

import lightgbm as lgb
from numba import jit

import petro2

# Parameters
LABEL_COLUMN_NAME = 'phi'
RANDOM_STATE = 1

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
    "random_state": 0
}

SEISMIC_MAX_X = 433
SEISMIC_MAX_Y = 645
SEISMIC_MAX_Z = 250


def eval_model(orig_df, main_df, features):

    # predict points marked for expand
    X_to_predict = main_df[main_df['real'] == 2]
    X_to_predict = X_to_predict[[petro2.f2str(f) for f in features]]

    # Only train on points with phi value
    X_with_phi = main_df[main_df['real'] != 2]
    X_with_phi = X_with_phi[[petro2.f2str(f) for f in features]]

    # Results (phi) only for predicted or original points
    y_df = main_df[main_df['real'] != 2][LABEL_COLUMN_NAME]

    # Train
    lgb_train = lgb.Dataset(X_with_phi.values, y_df.values)
    regressor = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
    )
    # callbacks=[lgb.log_evaluation(show_stdv=False)])

    # Predict expanded points
    pred = regressor.predict(X_to_predict.values)

    # Get the two disjoint set of points, real + previously expanded
    # and expanded on this iteration
    remaining_df = orig_df[orig_df['real'] != 2]
    predicted_df = orig_df[orig_df['real'] == 2]

    # Update real value from 2 (to expand) to 1 (expanded)
    predicted_df.loc[:, 'real'] = 1

    # Assign predicted values
    predicted_df.loc[:, 'phi'] = pred

    return pd.concat([remaining_df, predicted_df])


@jit(nopython=True)
def parallel_read(array_np, indexes, f_x, f_y, f_z):
    ret = np.empty((len(indexes)), dtype=np.float64)

    ii = 0
    for i in indexes:
        x = max(0, min(SEISMIC_MAX_X, i[0] + f_x))
        y = max(0, min(SEISMIC_MAX_Y, i[1] + f_y))
        z = max(0, min(SEISMIC_MAX_Z, i[2] + f_z))
        coord = x * (SEISMIC_MAX_Y + 1) * (SEISMIC_MAX_Z +
                                           1) + y * (SEISMIC_MAX_Z + 1) + z
        ret[ii] = array_np[coord]
        ii = ii + 1

    return ret


# Uses ndarray instead of pandas access
def get_feature_col2(indexes, feature, features_df):
    if type(feature) is tuple:
        sub_features_np = features_df[feature[0]].values

        # Numba only accepts ndarrays of concrete types (not object)
        indexes_ndarray = np.array(indexes.values,
                                   dtype=[('x', '<u2'), ('y', '<u2'),
                                          ('z', '<u2')])
        return parallel_read(sub_features_np, indexes_ndarray, feature[1],
                             feature[2], feature[3])
    else:
        return features_df[features_df.index.isin(indexes)][feature].values


def perf_predition(best_features_set, main_df, features_df):
    t1 = time.time()

    # Remove coordinates from features set
    best_features_set.remove('x')
    best_features_set.remove('y')
    best_features_set.remove('z')

    # Create a DataFrame for the features to be used for prediction
    cur_features_df = main_df.copy(deep=False)

    # Add each feature to the DataFrame
    for feature in best_features_set:
        print(f'[apply4] adding feature {feature}')
        feature_s = petro2.f2str(feature)
        cur_features_df.loc[:, feature_s] = get_feature_col2(
            main_df.index, feature, features_df)
    
    t2 = time.time()

    # print(cur_features_df)

    # Train model and predict porosity for new expanded points
    ret = eval_model(main_df, cur_features_df, best_features_set)
    t3 = time.time()

    print(f'[apply4] prep time: {t2-t1}, eval time: {t3-t2}')

    return ret
