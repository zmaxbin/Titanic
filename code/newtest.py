# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)

test = train_df['SibSp']

test_dummy = pd.get_dummies(test)

X = test_dummy
y = train_df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

model_Logistic = LogisticRegressionCV().fit(X_train, y_train)

y_pred = model_Logistic.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
parameter = model_Logistic.coef_
print parameter
