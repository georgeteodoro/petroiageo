import pandas as pd
import numpy as np
import random
import sys
import time
from io import StringIO

import lightgbm as lgb

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
    # main_df = main_df[main_df['phi'] != 2]

    # X = main_df[[petro2.f2str(f) for f in features]]

    # predict points marked for expand
    X_to_predict = main_df[main_df['real'] == 2]
    X_to_predict = X_to_predict[[petro2.f2str(f) for f in features]]

    # Only train on points with phi value
    X_with_phi = main_df[main_df['real'] != 2]
    X_with_phi = X_with_phi[[petro2.f2str(f) for f in features]]

    # Results (phi) only for predicted or original points
    y_df = main_df[main_df['real'] != 2][LABEL_COLUMN_NAME]

    print("training")
    lgb_train = lgb.Dataset(X_with_phi.values, y_df.values)
    regressor = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
    )
    # callbacks=[lgb.log_evaluation(show_stdv=False)])
    pred = regressor.predict(X_to_predict.values)

    predicted_df = orig_df[orig_df['real'] == 2]
    predicted_df.loc[:, 'real'] = 1
    predicted_df.loc[:, 'phi'] = pred

    remaining_df = orig_df[orig_df['real'] != 2]

    return pd.concat([remaining_df, predicted_df])


def get_feature_col(indexes, feature, features_df):
    if type(feature) is tuple:
        t1 = time.time()
        ret = np.empty(len(indexes))
        t2 = time.time()
        # Filtering earlier is faster than selecting col on .loc
        sub_features = features_df[feature[0]]
        ii = 0
        t3 = time.time()
        print(f'len {len(indexes)}')
        for i in indexes:
            # t31 = time.time()
            x = max(0, min(SEISMIC_MAX_X, i[0] + feature[1]))
            y = max(0, min(SEISMIC_MAX_Y, i[1] + feature[2]))
            z = max(0, min(SEISMIC_MAX_Z, i[2] + feature[3]))
            # t32 = time.time()
            ret[ii] = sub_features.loc[(x, y, z)]
            # t33 = time.time()
            # print(f'=== coord: {t2-t1}')
            # print(f'=== load: {t3-t2}')
            ii = ii + 1
        t4 = time.time()

        print(f'[get_feature_col]:')
        print(f'   np alloc: {t2-t1}')
        print(f'   filtering: {t3-t2}')
        print(f'   iteration: {t4-t3}')
        return ret

    else:
        return features_df[features_df.index.isin(indexes)][feature].values


# Uses ndarray instead of pandas access
def get_feature_col2(indexes, feature, features_df):
    if type(feature) is tuple:
        ret = np.empty(len(indexes))
        sub_features_np = features_df[feature[0]].values
        ii = 0
        for i in indexes:
            x = max(0, min(SEISMIC_MAX_X, i[0] + feature[1]))
            y = max(0, min(SEISMIC_MAX_Y, i[1] + feature[2]))
            z = max(0, min(SEISMIC_MAX_Z, i[2] + feature[3]))
            coord = x * (SEISMIC_MAX_Y + 1) * (SEISMIC_MAX_Z +
                                               1) + y * (SEISMIC_MAX_Z + 1) + z
            ret[ii] = sub_features_np[coord]
            ii = ii + 1

        return ret

    else:
        return features_df[features_df.index.isin(indexes)][feature].values


def perf_predition(best_features_set, main_df, features_df):

    # Create DataFrame with only the features
    best_features_set.remove('x')
    best_features_set.remove('y')
    best_features_set.remove('z')
    features_dic = dict()

    # Second approach is faster
    # filtered_features_df = features_df.filter(items=main_df.index.values,
    #                                           axis=0)
    # filtered_features_df = features_df[features_df.index.isin(main_df.index)]

    cur_features_df = main_df.copy(deep=False)

    for feature in best_features_set:
        print(f'[apply4] adding feature {feature}')
        feature_s = petro2.f2str(feature)
        t1 = time.time()
        tmp = get_feature_col2(main_df.index, feature, features_df)
        t2 = time.time()
        cur_features_df.loc[:, feature_s] = tmp
        t3 = time.time()
        print(f'   filtering: {t2-t1}, assignment: {t3-t2}')

    # print(cur_features_df)

    return eval_model(main_df, cur_features_df, best_features_set)
