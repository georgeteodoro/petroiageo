import pandas as pd
import numpy as np
import random
import sys

import lightgbm as lgb


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

def eval_model(df, features):
   X = df[features].values
   y = df[LABEL_COLUMN_NAME].values

   lgb_train = lgb.Dataset(X, y)
   regressor = lgb.train(params, lgb_train, verbose_eval=False, num_boost_round=100)
   pred = regressor.predict(X)
   for i in range(len(X)):
      if pred[i]<0: pred[i] = 0
      if df['real'][i]==1 or df['real'][i]==2: pred[i] = df[LABEL_COLUMN_NAME][i]
      #if df['real'][i]==1: pred[i] = df[LABEL_COLUMN_NAME][i]
      #elif df['real'][i]==2: pred[i] = (pred[i]+df[LABEL_COLUMN_NAME][i])/2
      print(int(X[i][0]), int(X[i][1]), int(X[i][2]), pred[i])

# Reads dataset
df = pd.read_csv(sys.argv[1])
df.dropna(axis=0, subset=[LABEL_COLUMN_NAME], inplace=True)

all_features = list(df.columns)
all_features.remove('X')
all_features.remove('Y')
all_features.remove('depth')
all_features.remove('well')
all_features.remove('real')
all_features.remove('rho')
all_features.remove('vp')
all_features.remove('vs')
all_features.remove(LABEL_COLUMN_NAME)
f = ['X', 'Y', 'depth', 'mid -1 -2 -2', 'gst 3 0 1', 'gst -2 2 -2', 'gst -1 -1 -2', 'gst -1 2 -1', 'gst 3 0 2', 'gst -3 -2 -1', 'gersz 3 0 -1', 'gst 1 2 2', 'mid -3 3 1'] #remover
eval_model(df, f) #remover
