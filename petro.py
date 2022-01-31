import pandas as pd
import numpy as np
import sys
import os
import random
from io import StringIO

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = 'phi'
UNWANTED_COLUMNS = ['real', 'well', 'rho', 'vs', 'vp']

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


def eval_bootstrap(df, features):
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values
    a = []
    b = []

    logo = LeaveOneGroupOut()
    groups = df['well']
    logo.get_n_splits(X, y, groups)
    logo.get_n_splits(groups=groups)
    for (train, val) in logo.split(X, y, groups):
        v = []
        n = 0
        for i in np.array(df['real'][val]):
            if i == 1:  # if REAL
                v.append(n)
            n = n + 1
        v = np.array(v)
        lgb_train = lgb.Dataset(X[train], y[train])
        #lgb_eval = lgb.Dataset(X[val], y[val], reference=lgb_train)
        lgb_eval = lgb.Dataset(X[v], y[v], reference=lgb_train)
        regressor = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_eval,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False)
            ])
        pred = regressor.predict(X[v])
        rmse = np.sqrt(np.mean((pred - y[v])**2))
        mae = mean_absolute_error(pred, y[v])
        a.append(rmse)
        b.append(mae)
    return np.mean(a), np.mean(b)


def get_features_sets(str_nwells):
    # Reads dataset
    str_nwells = StringIO(str_nwells)
    str_nwells.seek(0)
    df = pd.read_csv(str_nwells)
    df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

    RANDOM_STATE = 1
    all_features = list(df.columns)
    for x in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME]:
        all_features.remove(x)

    print("[petro] Starting features set search")
    f = ['X', 'Y', 'depth']
    i = 0
    results = []
    for f1 in all_features:
        if i == 10: break
        if f1 in f: continue
        k = 1000
        x = f1
        i = i + 1
        j = 0
        for f2 in all_features:
            if f2 in f: continue
            j = j + 1
            if j == 5: break  # FOR DEBUGING => less iterations
            f.append(f2)
            print(f)
            A, B = eval_bootstrap(df, f)
            # s.write(f"{f},{A},{B}")
            results.append([f, A, B])
            z = A
            f.remove(f2)
            sys.stdout.flush()
            if z < k:
                x = f2
                k = z
        f.append(x)

    return results
