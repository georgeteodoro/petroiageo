import pandas as pd
import numpy as np
import random
import sys
import time

from sklearn.model_selection import LeaveOneGroupOut
from sklearn import ensemble
import lightgbm as lgb

# Parameters
LABEL_COLUMN_NAME = 'phi'
RANDOM_STATE = 1


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


def predict_lightgbm(x, y):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        # 'device_type': 'gpu',
        'random_state': RANDOM_STATE
    }

    gbm = lgb.train(
        params,
        lgb.Dataset(x, y),
        num_boost_round=100,
        # valid_sets=lgb_eval,
        # early_stopping_rounds=2,
        valid_sets=lgb.Dataset(x, y))

    return gbm.predict(x, num_iteration=gbm.best_iteration)


# def eval_bootstrap(df, features):
#     X = df[features].values
#     y = df[LABEL_COLUMN_NAME].values
#     a = []
#     b = []

#     logo = LeaveOneGroupOut()
#     groups = df['well']
#     logo.get_n_splits(X, y, groups)
#     logo.get_n_splits(groups=groups)
#     for (train, val) in logo.split(X, y, groups):
#         v = []
#         n = 0
#         for i in np.array(df['real'][val]):
#             if i == 1 or i == 2: v.insert(len(v),n)
#             n = n + 1
#         v = np.array(v)
#         regressor = ensemble.GradientBoostingRegressor(n_estimators = 30, max_depth = 10, min_samples_split = 5, learning_rate = 0.1, loss = 'ls', random_state = RANDOM_STATE)
#         regressor = regressor.fit(X[train], y[train])
#         pred = regressor.predict(X[v])
#         rmse = np.sqrt(np.mean((pred - y[v])**2))
#         mae = mean_absolute_error(pred, y[v])
#         a.insert(len(a), rmse)
#         b.insert(len(b), mae)
#     return np.mean(a),np.mean(b)


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

eval_model(df, coord_labels + cur_features_labels)





# labels = ['X', 'Y', 'depth']
# i = 0
# for f1 in coord_labels + cur_features_labels:
#     if i == 15: break # no more than 15 features 
#     i = i + 1

#     if f1 in labels: continue
#     x = f1
    
#     max_error = 1000
    
#     for f2 in filtered_features:
#         if f2 in labels: continue
#         rmse, mae = eval_bootstrap(df, labels + [f2])
#         print("%s,%f,%f" % (labels + [f2], rmse, mae))
#         sys.stdout.flush()
#         if rmse < max_error:
#             x = f2
#             max_error = rmse
#     labels.append(x)

