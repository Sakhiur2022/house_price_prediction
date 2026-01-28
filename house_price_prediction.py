import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('house_price_bd.csv')

data

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