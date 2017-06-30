# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)

df_train = train_df.drop('Cabin',1)
df_test = test_df.drop('Cabin',1)

df_train = df_train.drop('Name',1)
df_test = df_test.drop('Name',1)

# df_train = df_train.drop('Ticket',1)
# df_test = df_test.drop('Ticket',1)

all_data = pd.concat((df_train.loc[:,'Pclass':'Embarked'],
                      df_test.loc[:,'Pclass':'Embarked']))

size_mapping1 = {
    'C': 3,
    'Q': 2,
    'S': 1
}
all_data['Embarked'] = all_data['Embarked'].map(size_mapping1)
#
size_mapping2 = {
    'male': 1,
    'female': 0
}
all_data['Sex'] = all_data['Sex'].map(size_mapping2)

all_data_dummy = pd.get_dummies(all_data)
final_all_data = all_data_dummy.fillna(all_data_dummy.mean())

X_train = final_all_data[:df_train.shape[0]].values
X_test = final_all_data[df_train.shape[0]:].values
y = df_train.Survived.values

# model_Logistic = LogisticRegression(C=10,random_state=0).fit(X_train, y)

# model_SVC = SVC(kernel='linear',C=10.0,random_state=0).fit(X_train,y)

# y_pred_Logistic = model_Logistic.predict(X_test)
# y_pred_SVC = model_SVC.predict(X_test)

# y_final = (y_pred_Logistic+y_pred_SVC)/2

model_decisiontree = DecisionTreeClassifier().fit(X_train, y)
y_pred_Decisiontree = model_decisiontree.predict(X_test)

model_Randomforest = RandomForestClassifier(n_estimators=100).fit(X_train, y)
# y_pred_Randomforest = model_Randomforest.predic

submission_df = pd.DataFrame(data = {'PassengerId':test_df.index,'Survived':y_pred_Decisiontree})
submission_df.to_csv('../data/submission.csv',columns = ['PassengerId','Survived'],index = False)


