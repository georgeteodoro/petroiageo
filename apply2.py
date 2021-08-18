import pandas as pd
import numpy as np
import random
import sys

from sklearn import ensemble

# Parameters
LABEL_COLUMN_NAME = 'phi'
RANDOM_STATE = 1

def eval_model(df, features):
   X = df[features].values
   y = df[LABEL_COLUMN_NAME].values

   #regressor = ensemble.GradientBoostingRegressor(n_estimators = 30, max_depth = 10, min_samples_split = 5, learning_rate = 0.1, loss = 'ls', random_state = RANDOM_STATE)
   regressor = ensemble.GradientBoostingRegressor(n_estimators = 20, max_depth = 8, min_samples_split = 5, learning_rate = 0.1, loss = 'ls', random_state = RANDOM_STATE)
   regressor = regressor.fit(X, y)
   pred = regressor.predict(X)
   for i in range(len(X)):
      if pred[i]<0.025: pred[i] = 0
      if df['real'][i]==1 or df['real'][i]==2: pred[i] = df[LABEL_COLUMN_NAME][i]
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

f = ['X', 'Y', 'depth']
for x in all_features:
   f.insert(len(f), x)

eval_model(df, f)
