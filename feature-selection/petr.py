import pandas as pd
import numpy as np
import sys
import os
import random
from time import time
from enum import Enum

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = 'phi'
UNWANTED_COLUMNS = ['real', 'well', 'rho', 'vs', 'vp']

RANDOM_STATE = 0
MAX_FEATURES = 10
REAL_ID = 1

# MPI message tags
WORKER = Enum(SEND_RESULT, REQUEST_TASK)
MANAGER = Enum(NEW_TASK, DONE, NEW_FEATURE_BCAST)

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
    "random_state": RANDOM_STATE
}


def eval_bootstrap(df, features):
    X = df[features].values
    y = df[LABEL_COLUMN_NAME].values
    rmse_list = []
    mae_list = []

    logo = LeaveOneGroupOut()
    groups = df['well']
    logo.get_n_splits(X, y, groups)
    logo.get_n_splits(groups=groups)

    m=0

    for (train, val) in logo.split(X, y, groups):
        m = m+1
        n = 0
        real_vals_ids = []
        for i in np.array(df['real'][val]):
            if i == REAL_ID: real_vals_ids.append(n)
            n = n + 1
        real_vals_ids = np.array(real_vals_ids)

        lgb_train = lgb.Dataset(X[train], y[train])
        lgb_eval = lgb.Dataset(X[real_vals_ids],
                               y[real_vals_ids],
                               reference=lgb_train)
        regressor = lgb.train(params,
                              lgb_train,
                              verbose_eval=False,
                              num_boost_round=100,
                              valid_sets=lgb_eval,
                              early_stopping_rounds=30)
        pred = regressor.predict(X[real_vals_ids])

        rmse = np.sqrt(np.mean((pred - y[real_vals_ids])**2))
        mae = mean_absolute_error(pred, y[real_vals_ids])

        rmse_list.append(rmse)
        mae_list.append(mae)
    print(m)

    return np.mean(rmse_list), np.mean(mae_list)


# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

cur_features = ['X', 'Y', 'depth']
all_features = list(df.columns)
for x in UNWANTED_COLUMNS + [LABEL_COLUMN_NAME] + cur_features:
    all_features.remove(x)

f = open('petr-times.log', mode='w')
print("Training " + str(MAX_FEATURES) + " out of " +
      str(len(all_features)) + " total features",
      file=f)
print("Data size: " + str(len(df)), file=f)

# for i in range(MAX_FEATURES):
#     t1 = time()
    
#     # Reset best_feature
#     max_error = 100000
#     best_feature = None

#     train_times = []

#     # Find the best feature for the current iteration i
#     for cur_feature in all_features:
#         t2 = time()

#         rmse, mae = eval_bootstrap(df, cur_features + [cur_feature])
#         print("%s,%f,%f" % (cur_features + [cur_feature], rmse, mae))
#         sys.stdout.flush()

#         # Updates best feature
#         if mae < max_error:
#             max_error = mae
#             best_feature = cur_feature

#         t3 = time()
#         train_times.append(t3 - t2)

#     # Commit best feature from this iteration and remove it from the
#     cur_features.append(best_feature)
#     all_features.remove(best_feature)

#     t4 = time()

#     print("it[" + str(i) + "] => it_time=" + str(t4 - t1) +
#           " - avg_train_time=" + str(np.average(train_times)),
#           file=f)


# Initialize MPI processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
manager_rank = comm.Get_size()-1
num_workers = comm.Get_size()-1

if rank == manager_rank:
    manager()
else:
    worker()


def manager():
    # Test MAX_FEATURES features, adding them to cur_features
    for it in range(MAX_FEATURES):

        # Reset variables of the current feature iteration
        available_features = all_features.copy()
        cur_results = []
        workers_done = 0

        # MPI messages loop
        while True:
            status = MPI.Status()
            msg = comm.recv(status=status)
            worker_id = status.Get_source()
            
            match status.Get_tag():
                case WORKER.SEND_RESULT:
                    # Results from the worker are only appended to the
                    # current results list, for later selection
                    cur_results.append(msg)

                case WORKER.REQUEST_TASK:
                    # If there are no more features to be calculated, the 
                    # worker must prepare to receive the best feature from
                    # this iteration
                    if available_features.empty():
                        comm.send(data=[], 
                                  tag=MANAGER.NEW_FEATURE_BCAST, 
                                  dest=worker_id)
                        
                        # Check if all workers have finished their tasks
                        workers_done = workers_done + 1
                        if workers_done == num_workers:
                            break
                    else:
                        # Send a feature for evaluation on a worker
                        feature = available_features.pop()
                        comm.send(MANAGER.NEW_TASK, feature)

        # Reset best_feature
        max_error = 100000
        best_feature = None

        # Find best feature
        cur_results.sort() # for validation: all results are the same
        for (feature, rmse, mae) in cur_results:
            print("%s,%f,%f" % (cur_features + [feature], rmse, mae))
            sys.stdout.flush()

            # Updates best feature
            if mae < max_error:
                max_error = mae
                best_feature = cur_feature

        # Commit best feature from this iteration
        comm.bcast(best_feature)
        cur_features.append(best_feature)
        all_features.remove(best_feature)

    # Kill all workers
    for i in range(num_workers):
        comm.recv()
        comm.send(MANAGER.DONE)

def worker():
    while true:
        comm.send(WORKER.REQUEST_TASK)
        msg = comm.recv()

        match msg.tag:
            case MANAGER.DONE:
                break

            case MANAGER.NEW_FEATURE_BCAST:
                comm.bcast(new_feature)
                cur_features.append(new_feature)

            case MANAGER.NEW_TASK:
                feature = msg.value
                rmse, mae = eval_bootstrap(df, cur_features + [feature])

                comm.send(WORKER.SEND_RESULT, (feature, rmse, mae))





