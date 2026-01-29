import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('house_price_bd.csv')

data.info()

data.dropna(inplace=True)

data.drop(['title','block/sector','city/area'],axis=1,inplace=True)

data.info()

from sklearn.model_selection import train_test_split

X = data.drop(['price'],axis=1)

y = data['price']

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.8)

train_data=X_train.join(y_train)

sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")

train_data['price']=np.log(train_data['price']+1)

train_data.hist()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

reg = LinearRegression()
X_train,y_train=train_data.drop(['price'],axis=1),train_data['price']
X_train_s=StandardScaler().fit_transform(X_train)
reg.fit(X_train,y_train)

test_data=X_test.join(y_test)
test_data['price']=np.log(test_data['price']+1)

X_test,y_test=test_data.drop(['price'],axis=1),test_data['price']
X_test_s=StandardScaler().fit_transform(X_test)
reg.score(X_test,y_test)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(X_train,y_train)
forest.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV
forest=RandomForestRegressor()
param_grid={'n_estimators':[10,20,30,40,50,60,70,80,90,100],
            'max_depth':[1,2,3,4,5,6,7,8,9,10],
            'max_features':[1,2,3,4] }
grid_search=GridSearchCV(forest,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_test_s,y_test)

best_forest=grid_search.best_estimator_
best_forest.score(X_test_s,y_test)
