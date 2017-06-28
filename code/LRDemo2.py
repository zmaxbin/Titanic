# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)

train_df.drop('Cabin',1)
test_df.drop('Cabin',1)

train_df.drop('Name',1)
test_df.drop('Name',1)

train_df.drop('Ticket',1)
test_df.drop('Ticket',1)

# size_mapping1 = {
#     'C': 1,
#     'Q': 2,
#     'S': 3
# }
# train_df['Embarked'] = train_df['Embarked'].map(size_mapping1)
#
# size_mapping2 = {
#     'male': 1,
#     'female': 0
# }
# train_df['Sex'] = train_df['Sex'].map(size_mapping2)

train_df = pd.get_dummies(train_df)
train_df = train_df.fillna(train_df.mean())

X = train_df.iloc[:,:-1].values
y = train_df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

weight = []
params = []
for c in [0.01,0.1,1,10,20,30,50,80,100,200,500,1000]:
    lr = SVC(kernel='linear',C=c,random_state=0)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    print 'Accuracy:%.2f' %accuracy_score(y_test,y_pred)