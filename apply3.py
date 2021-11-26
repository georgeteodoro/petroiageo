import pandas as pd
import numpy as np

from random import random
import sys
import time
from math import ceil
from termcolor import colored

from mpi4py import MPI

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import ensemble
import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = 'phi'
RANDOM_STATE = 1


def printm(string):
    print(colored("[Manager] ", "yellow") + string)


def printw(string, id=0):
    print(colored("[Worker][" + str(id) + "] ", "green") + string)


def get_error(df, pred):
    mae = 0
    # rmsd = 0
    count = 0
    for i in range(len(pred)):
        if df['real'][i] == 1:  # TODO: optimize 'if' out with products
            mae += abs(pred[i] - df[LABEL_COLUMN_NAME][i])
            # rmsd += (pred[i] - df[LABEL_COLUMN_NAME][i])**2
            count += 1
    if count == 0:
        return 0
    else:
        # return mae / count, np.sqrt(rmsd / count)
        return mae / count


def predict_gradient_boosting(x, y):
    regressor = ensemble.GradientBoostingRegressor(n_estimators=20,
                                                   max_depth=8,
                                                   min_samples_split=5,
                                                   learning_rate=0.1,
                                                   loss='ls',
                                                   random_state=RANDOM_STATE)

    regressor = regressor.fit(x, y)
    return regressor.predict(x)


def predict_lightgbm(xTrain, yTrain, xReal):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        # 'device_type': 'gpu',
        'random_state': RANDOM_STATE
    }

    gbm = lgb.train(
        params,
        lgb.Dataset(xTrain, yTrain),
        num_boost_round=100,
        verbose_eval=False,
        # valid_sets=lgb_eval,
        # early_stopping_rounds=2,
        valid_sets=lgb.Dataset(xTrain, yTrain))

    return gbm.predict(xReal, num_iteration=gbm.best_iteration)


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
            if i == 1 or i == 2: v.insert(len(v), n)
            n = n + 1
        v = np.array(v)
        pred = predict_lightgbm(X[train], y[train], X[v])
        rmse = np.sqrt(np.mean((pred - y[v])**2))
        mae = mean_absolute_error(pred, y[v])
        a.insert(len(a), rmse)
        b.insert(len(b), mae)
    return np.mean(a), np.mean(b)


def mock(a, b):
    return random(), random()


# OLD without feature selection
def eval_model(df, features):
    t1 = time.time()
    x = df[features].values
    y = df[LABEL_COLUMN_NAME].values

    t2 = time.time()
    # pred = predict_gradient_boosting(x, y)
    pred = predict_lightgbm(x, y)

    t3 = time.time()
    mae = get_error(df, pred)

    t4 = time.time()
    for i in range(len(x)):
        if pred[i] < 0.025: pred[i] = 0
        # Label=1 means real well point
        # Label=2 means predicted point of previous iterations
        if df['real'][i] == 1 or df['real'][i] == 2:
            pred[i] = df[LABEL_COLUMN_NAME][i]
        print(int(x[i][0]), int(x[i][1]), int(x[i][2]), pred[i])

    t5 = time.time()

    with open('apply-error.log', mode='a') as f:
        print("MAE: " + str(mae), file=f)

    with open('apply-times.log', mode='a') as f:
        print("prep: " + str(t2 - t1), file=f)
        print("predict: " + str(t3 - t2), file=f)
        print("error-calc: " + str(t4 - t3), file=f)
        print("write: " + str(t5 - t4), file=f)
        print("", file=f)


# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

cur_features_labels = list(df.columns)
cur_features_labels.remove('well')
cur_features_labels.remove('real')
cur_features_labels.remove('X')
cur_features_labels.remove('Y')
cur_features_labels.remove('depth')
cur_features_labels.remove(LABEL_COLUMN_NAME)
cur_features_labels.remove('rho')
cur_features_labels.remove('vp')
cur_features_labels.remove('vs')

coord_labels = ['X', 'Y', 'depth']

# Current best set of labels
cur_labels = coord_labels

# TODO: set this list
filtered_features = cur_features_labels

# Initialize MPI vars
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
workers_size = mpi_size - 1
MANAGER_RANK = mpi_size - 1
RESULTS_TAG = 14
MAX_FEATURES = 1

i = 0
if rank == MANAGER_RANK:
    while len(filtered_features) > 0 and i < MAX_FEATURES:
        printm("Running i " + str(i))
        i = i + 1
        cur_error = 100000
        cur_feature = ''

        # Get all results
        for j in range(len(filtered_features)):
            # print("manager waiting " + str(j))
            feature, errors = comm.recv(source=MPI.ANY_SOURCE, tag=RESULTS_TAG)
            # print("manager received " + str(feature))

            # Compare features for best result
            if errors[1] < cur_error:
                cur_feature = feature

        # Broadcast best feature
        comm.bcast(feature, root=MANAGER_RANK)

        # Update local results
        filtered_features.remove(cur_feature)
        cur_labels.append(cur_feature)

else:
    while len(filtered_features) > 0 and i < MAX_FEATURES:
        printw("Running i " + str(i), rank)
        i = i + 1

        # Features evaluated locally, mapped by feature name
        evaluated = dict()

        t0 = time.time()

        # Evaluate local features
        features_per_rank = ceil(len(filtered_features) / workers_size)
        for j in range(features_per_rank):
            t1 = time.time()

            # Get index for current rank
            ii = rank + j * workers_size
            
            printw("Evaluating " + str(ii) + "/" + str(len(filtered_features)),
                   rank)

            # Verify for out-of-bounds cases
            if ii >= len(filtered_features):
                # print("worker continued")
                continue

            # Evaluate feature and store it
            cur_feature = filtered_features[j]
            rmse, mae = eval_bootstrap(df, cur_labels + [cur_feature])
            # rmse, mae = mock(df, cur_labels + [cur_feature])
            evaluated[cur_feature] = (rmse, mae)
            t2 = time.time()

            printw(
                "Evaluated " + str(ii) + "/" + str(len(filtered_features)) +
                " in " + str(t2 - t1) + " secs", rank)

        # Send results
        for e in evaluated.items():
            # print("worker send " + str(e))
            comm.send(e, dest=MANAGER_RANK, tag=RESULTS_TAG)

        # Wait for new feature
        new_feature = ''
        new_feature = comm.bcast(new_feature, root=MANAGER_RANK)
        # print("worker bcast received " + new_feature)
        filtered_features.remove(new_feature)
        cur_labels.append(new_feature)

        t3 = time.time()

if rank == MANAGER_RANK:
    # Print results of best feature set
    eval_model(df, cur_labels)
