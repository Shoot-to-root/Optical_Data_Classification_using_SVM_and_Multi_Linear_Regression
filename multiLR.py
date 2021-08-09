# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def pearson_coeff(x, y):
    return np.corrcoef(x, y)

""" review/preprocess dataset """
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)

data = pd.read_csv("dataset.csv", index_col=False)

#print(data.dtypes)
data = data.fillna(data.mean())
#print(data.isnull().sum())

""" handling categorical features """
#print(data["feature_2"].value_counts())
#print(data["feature_4"].value_counts())
convert_categorical = {"feature_2": {"F": 1, "M": 0},
                       "feature_4": {"OS": 1, "OD": 0}}
data = data.replace(convert_categorical)
#print(data.head())

""" split label """
label = data.label
data = data.drop(["label"], axis=1)
model = LinearRegression(n_jobs=-1, normalize=True)

""" 2(a) multiple linear regression """
result = pd.DataFrame(columns=["mul_LR_score"], index=None)
model.fit(data, label)
result.loc[0] = model.score(data, label)
print(result)
result.to_csv("mulLR_score.csv")

""" 2(b) calculate the simple linear regression for each input variable and output. 
And then rank them according training root mean square error """
result = pd.DataFrame(np.zeros(119), columns=["RMSE"], index=data.columns)
for i in range(119):
    feature = data.loc[:, "feature_{}".format(i)].values.reshape(-1, 1)
    model.fit(feature, label)
    RMSE = rmse(label, model.predict(feature)) 
    result.iloc[i, 0] = RMSE
    
#print(result)
result.to_csv("RMSE_score.csv")

""" 2(c) calculate the Pearson's correlation coefficient and rank according the values of correlation coefficients """
result = pd.DataFrame(np.zeros(119), columns=["PEARSON"], index=data.columns)
for i in range(119):
    feature = data["feature_{}".format(i)]
    pearson = pearson_coeff(feature, label) 
    result.iloc[i, 0] = pearson[0][1]
    
print(result)
result.to_csv("pearson_score.csv")
"""
for i in range(0, 119):
    for j in range(1, 119):  
        if i == j:
            pass
        else: 
            f1 = data["feature_{}".format(i)]
            f2 = data["feature_{}".format(j)]
            q2c = pearson(f1, f2)
            print("feature_{} -- feature_{}".format(i, j))
"""            

""" 2(e) find the suitable input variables to minimize the regression RMSE """
sorted_rmse = pd.read_csv("RMSE_score.csv", index_col=0)
result = pd.DataFrame(np.zeros(118), columns=["RMSE"], index=range(1, 119))
n_feature = []
for i in range(1,119):
    n_feature = sorted_rmse.index[0:i+1].to_list()
    feature = data.loc[:, n_feature]
    model.fit(feature, label)
    RMSE =rmse(label, model.predict(feature))
    result.iloc[i-1, 0] = RMSE
    #print(i, RMSE)
    
print(result)
result.to_csv("2e.csv")
