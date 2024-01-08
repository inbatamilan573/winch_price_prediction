# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:52:15 2024

@author: Inbat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('C:/Users/Inbat/Downloads/Winch Price Prediction Project-20240106T075431Z-001/Winch Price Prediction Project/winch_data.xlsx')

df.head()

df.shape

df.isnull().sum()

df.rename(columns={'Total weight for winch including all steel and all components':'Total weight', "Number of kilos of steel Weight of steel produced according to Seaonic's drawings":'weight of steel produced',"Weight of components purchased, gear, motor, fastening material etc.":'weight of components purchased',"Number of drawings with parts that need machining. There may be several numbers of the same part to be produced":'drawings with parts',"Number of drawings in the production material, including assemblies":'drawings in production material',"Number of machined parts that are in bill-of-material":'machine parts in bill-of-material',"Total number of parts for the winch both machined parts and purchased parts":'Total machined and purchased parts'},inplace=True)

df.drop(['Winch name','Dim Sketch','Producer'],axis=1, inplace=True)

df.columns

sns.pairplot(df.iloc[:,[1,2,3,17]])

plt.figure(figsize = (16,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


df.corr()['Total price NOK'].sort_values(ascending=False)


X = df.iloc[:,[0,3,4,6,12,16]]
y = df['Total price NOK']



cTransformer = ColumnTransformer([('encoder',OneHotEncoder(drop='first'),[0])],remainder='passthrough')

cTransformer.fit(X)

X = np.array(cTransformer.transform(X),dtype=np.float32)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



def validate_result(model,model_name):
    y_pred = model.predict(X_test)
    RMSE_score = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    print('Model: ', model_name)
    print('RMSE: ',RMSE_score)
    
    R2_score = metrics.r2_score(y_test,y_pred)
    print('R2 Score: ',R2_score)
    
    
import xgboost as xgb
xgboost_reg = xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=1000, verbosity=3)

xgboost_reg.fit(X_train,y_train)

validate_result(xgboost_reg,'XGBoost')


xgb.plot_importance(xgboost_reg)


import pickle
file = open("pickle_model_two.pkl","wb")
pickle.dump(xgboost_reg,file)


file_tranformer = open("pickle_cTransformer_two.pkl","wb")
pickle.dump(cTransformer,file_tranformer)



