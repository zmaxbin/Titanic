# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)

train_df.drop('Cabin',1)
test_df.drop('Cabin',1)

train_df.drop('Name',1)
test_df.drop('Name',1)

train_df.drop('Ticket',1)
test_df.drop('Ticket',1)

all_data = pd.concat((train_df.loc[:,'Pclass':'Embarked'],
                      test_df.loc[:,'Pclass':'Embarked']))

size_mapping1 = {
    'C': 3,
    'Q': 2,
    'S': 1
}
all_data['Embarked'] = all_data['Embarked'].map(size_mapping1)

size_mapping2 = {
    'male': 1,
    'female': 0
}
all_data['Sex'] = all_data['Sex'].map(size_mapping2)

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train_df.shape[0]]
X_test = all_data[train_df.shape[0]:]
y = train_df.Survived

model_Logistic = LogisticRegressionCV().fit(X_train, y)

model_SVC = SVC(kernel='linear',C=10.0,random_state=0).fit(X_train,y)

y_pred_Logistic = model_Logistic.predict(X_test)
y_pred_SVC = model_SVC.predict(X_test)

y_final = (y_pred_Logistic+y_pred_SVC)/2

submission_df = pd.DataFrame(data = {'PassengerId':test_df.index,'Survived':y_final})
submission_df.to_csv('../data/submission1.csv',columns = ['PassengerId','Survived'],index = False)


