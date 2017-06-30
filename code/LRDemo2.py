# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import BaggingClassifier

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)

# print(train_df.describe())

train_df = train_df.drop(['Cabin','Name'],axis=1)


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

df_dummy_train = pd.get_dummies(train_df)
final_train_df = df_dummy_train.fillna(df_dummy_train.mean())

# scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(final_train_df['Age'])
# final_train_df['Age'] = scaler.fit_transform(final_train_df['Age'], age_scale_param)
# fare_scale_param = scaler.fit(final_train_df['Fare'])
# final_train_df['Fare'] = scaler.fit_transform(final_train_df['Fare'], fare_scale_param)

X = final_train_df.iloc[:,2:].values
y = final_train_df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# for c in [0.1,1,10,20,50,100,500]:
#     lr = SVC(kernel='linear',C=c)
#     lr.fit(X_train,y_train)
#     y_pred = lr.predict(X_test)
#     print('Misclassified samples: %d' % (y_test != y_pred).sum())

# lr = LogisticRegressionCV().fit(X_train,y_train)
# y_pred = lr.predict(X_test)
model_decisiontree = DecisionTreeClassifier(max_depth=15).fit(X_train, y_train)
y_pred_Decisiontree = model_decisiontree.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred_Decisiontree).sum())

# error = 0
# for i in range(10):
#     decision_tree = DecisionTreeClassifier()
#     decision_tree.fit(X_train, y_train)
#     Y_pred = decision_tree.predict(X_test)
#     error_num = (y_test != Y_pred).sum()
#     error = error + error_num
# print(error/10)
# bagging_clf = BaggingClassifier().fit(X_train, y_train)
# y_pred = bagging_clf.predict(X_test)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())


# import xgboost as xgb
#
# error = 0
# for i in range(10):
#     gbm = xgb.XGBClassifier().fit(X_train,y_train)
#     predictions = gbm.predict(X_test)
#     error_num = (y_test != predictions).sum()
#     error = error + error_num
# print(error/10)

