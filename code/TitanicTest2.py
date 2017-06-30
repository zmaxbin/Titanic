# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)

train_df['Embarked'] = train_df['Embarked'].fillna('C')

fare_median = test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')]['Fare'].median()
test_df['Fare'] = test_df['Fare'].fillna(fare_median)

train_df['Deck'] = train_df['Cabin'].str[0]
test_df['Deck'] = test_df['Cabin'].str[0]

train_df.ix[(train_df['Pclass']==1) & (train_df['Deck'].isnull() == True),'Deck'] = 'C'

train_df['Deck'] = train_df['Deck'].fillna('F')

train_df['Family'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df.loc[train_df['Family'] == 1, 'FamilyType'] = 'singleton'
train_df.loc[(train_df['Family'] > 1) & (train_df['Family'] < 5), 'FamilyType'] = 'small'
train_df.loc[train_df['Family'] > 4, 'FamilyType'] = 'large'

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=['Embarked','Sex',"FamilyType",'Deck','Ticket']
for col in cat_vars:
    train_df[col]=labelEnc.fit_transform(train_df[col])

from sklearn.ensemble import RandomForestRegressor


def fill_missing_age(data):
    # Feature set
    features = data[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp',
                     'Pclass','Deck']]
    # Split sets into train and prediction
    train = features.loc[(data.Age.notnull())]  # known Age values
    prediction = features.loc[(data.Age.isnull())]  # null Ages

    # All age values are stored in a target array
    y = train.values[:, 0]

    # All the other values are stored in the feature array
    X = train.values[:, 1::]

    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)

    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(prediction.values[:, 1::])

    # Assign those predictions to the full data set
    data.loc[(data.Age.isnull()), 'Age'] = predictedAges

    return data

train_df = fill_missing_age(train_df)

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(train_df[['Age', 'Fare']])
train_df[['Age', 'Fare']] = std_scale.transform(train_df[['Age', 'Fare']])


std_scale = preprocessing.StandardScaler().fit(train_df[['Age', 'Fare']])

y = train_df.Survived

train_df = train_df.drop(['Cabin','Name','PassengerId','Survived','SibSp','Parch','Family'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train_df,y,test_size=0.3,random_state=0)


model_decisiontree = LogisticRegressionCV().fit(x_train, y_train)
y_pred_Decisiontree = model_decisiontree.predict(x_test)
print('Misclassified samples: %d' % (y_test != y_pred_Decisiontree).sum())

# colormap = plt.cm.viridis
# plt.figure(figsize=(12,12))
# plt.title('Feature correlations', y=1.05, size=15)
# sns.heatmap(train_df.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

