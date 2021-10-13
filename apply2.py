import pandas as pd
import numpy as np
import random
import sys
import time

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


# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

all_features = list(df.columns)
all_features.remove('well')
all_features.remove('real')
all_features.remove('X')
all_features.remove('Y')
all_features.remove('depth')
all_features.remove(LABEL_COLUMN_NAME)
all_features.remove('rho')
all_features.remove('vp')
all_features.remove('vs')

f = ['X', 'Y', 'depth']
for x in all_features:
    f.insert(len(f), x)

eval_model(df, f)
