import numpy as np
import pandas as pd
import re as re
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)
full_data = [train_df, test_df]

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('C')

for dataset in full_data:
    dataset['NiceParch'] = 0
    dataset.loc[(dataset['Parch'] < 4) & (dataset['Parch'] > 0), 'NiceParch'] = 1
# print (train_df[['NiceParch', 'Survived']].groupby(['NiceParch'], as_index=False).mean())

for dataset in full_data:
    dataset['NiceSibSp'] = 0
    dataset.loc[(dataset['SibSp'] < 3) & (dataset['SibSp'] > 0), 'NiceSibSp'] = 1
# print (train_df[['NiceSibSp', 'Survived']].groupby(['NiceSibSp'], as_index=False).mean())

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(test_df['Fare'].median())

for dataset in full_data:
    dataset['Age'] = dataset['Age'].fillna(train_df['Age'].median())

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset.loc[dataset['Age'] <= 18, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 64), 'Age'] = 1
    dataset.loc[dataset['Age'] > 64, 'Age'] = 2

# Feature Selection
drop_elements = ['Survived','PassengerId','Ticket', 'Name', 'Cabin', 'SibSp', \
                 'Parch']
drop_elements1 = ['Name', 'Cabin', 'Ticket','SibSp', \
                 'Parch']

train = train_df.drop(drop_elements, axis=1)

test = test_df.drop(drop_elements1, axis=1)

all_data = pd.concat((train,test))

# all_data_dummy = pd.get_dummies(all_data)

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train_df.Survived

print(X_train.head(10))



x_train, x_test, y_train, y_test = train_test_split(X_train,y,test_size=0.3,random_state=0)


model_decisiontree = DecisionTreeClassifier().fit(x_train, y_train)
y_pred_Decisiontree = model_decisiontree.predict(x_test)
print('Misclassified samples: %d' % (y_test != y_pred_Decisiontree).sum())


# submission_df = pd.DataFrame(data = {'PassengerId':test_df.index,'Survived':y_pred_Decisiontree})
# submission_df.to_csv('../data/submission2.csv',columns = ['PassengerId','Survived'],index = False)