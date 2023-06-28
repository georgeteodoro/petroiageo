import pandas as pd
import numpy as np
import random
import sys
from io import StringIO

import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = "phi"
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
    "random_state": 0,
}


def eval_model(df, features):
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values

    lgb_train = lgb.Dataset(X, y)
    regressor = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
    )
    # callbacks=[lgb.log_evaluation(show_stdv=False)])
    pred = regressor.predict(X)
    results = []
    for i in range(len(X)):
        if pred[i] < 0:
            pred[i] = 0
        if df["real"][i] == 1 or df["real"][i] == 2:
            pred[i] = df[LABEL_COLUMN_NAME][i]
        # if df['real'][i]==1: pred[i] = df[LABEL_COLUMN_NAME][i]
        # elif df['real'][i]==2: pred[i] = (pred[i]+df[LABEL_COLUMN_NAME][i])/2
        # print(int(X[i][0]), int(X[i][1]), int(X[i][2]), pred[i])
        results.append([int(X[i][0]), int(X[i][1]), int(X[i][2]), pred[i]])

    return results


def perf_predition(f, str_nwells):
    # Reads dataset
    str_nwells = StringIO(str_nwells)
    str_nwells.seek(0)
    df = pd.read_csv(str_nwells)
    df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

    all_features = list(df.columns)
    all_features.remove("X")
    all_features.remove("Y")
    all_features.remove("depth")
    all_features.remove("well")
    all_features.remove("real")
    all_features.remove("rho")
    all_features.remove("vp")
    all_features.remove("vs")
    all_features.remove(LABEL_COLUMN_NAME)
    print(f)
    return eval_model(df, f)
