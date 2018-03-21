# Import libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
reg = linear_model.Ridge(alpha=0.1)

# Import data
df = pd.read_csv('train.csv') #header=None
#print(df.keys)
Y=df['y']
X=df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]

kf = KFold(n_splits=10)
kf.split(X, Y)
result = 0
for train_index, test_index in kf.split(X):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_train = Y.iloc[train_index]
    Y_test = Y.iloc[test_index]
    reg.fit(X_train,Y_train)
    Y_pred = reg.predict(X_test) #change this to X_test
    RMSE = mean_squared_error(Y_test, Y_pred)**0.5 #change this to Y_test
    result = result + RMSE
result = result/10
    