# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df  = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv",index_col=0)


# df_dummy_train = pd.get_dummies(train_df)
#
# # 画出相关系数图
# colormap = plt.cm.viridis
# plt.figure(figsize=(12,12))
# plt.title('Feature correlations', y=1.05, size=15)
# sns.heatmap(df_dummy_train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

# fare_median = test_df[(test_df['Pclass'] == 3) & (test_df['Embarked'] == 'S')]['Fare'].median()
# test_df['Fare'] = test_df['Fare'].fillna(fare_median)
# print(fare_median)
#

# 获取属性值的第一个字符
train_df['Deck'] = train_df['Cabin'].str[0]
test_df['Deck'] = test_df['Cabin'].str[0]

# 将Pclas为1，Embarked为NaN的值筛选出来，并且赋予新值C
train_df.ix[(train_df['Pclass']==1) & (train_df['Embarked'].isnull() == True),'Deck'] = 'C'

train_df['Deck'] = train_df['Deck'].fillna('F')

size_mapping = {
    'C': 0,
    'E': 1,
    'G':2,
    'D':3,
    'A':4,
    'B':5,
    'F':6,
    'T':7
}
train_df['Deck'] = train_df['Deck'].map(size_mapping)

# 画出三者的关系箱形图
sns.boxplot(x="Deck", y="Fare", hue="Pclass", data=train_df)
plt.show()

# 查看船舱
print(train_df['Deck'].unique())
# 画出在 Deck为NaN的情况下Fare的直方图
(train_df[train_df['Deck'].isnull()]['Fare']).hist(bins=100)
plt.show()


# 可以看出票的级别越高（数字越小）,存活概率越大
# print train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
   # Pclass  Survived
# 0       1  0.629630
# 1       2  0.472826
# 2       3  0.242363

# 可以看出女性的存活率高于男性
size_mapping = {
    'male': 0,
    'female': 1
}
train_df['Sex'] = train_df['Sex'].map(size_mapping)
# print train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#    Sex  Survived
#0    0  0.742038
# 1    1  0.188908

# 画出年龄与存活图
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# 携带的兄弟(姐妹)/配偶数量与存活的关系
# print train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
   # SibSp  Survived
# 1      1  0.535885
# 2      2  0.464286
# 0      0  0.345395
# 3      3  0.250000
# 4      4  0.166667
# 5      5  0.000000
# 6      8  0.000000

# 携带的父母/孩子数量与存活的关系
# print train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
   # Parch  Survived
# 3      3  0.600000
# 1      1  0.550847
# 2      2  0.500000
# 0      0  0.343658
# 5      5  0.200000
# 4      4  0.000000
# 6      6  0.000000

# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Fare', bins=100)
# plt.show()


size_mapping = {
    'C': 1,
    'Q': 2,
    'S': 3
}
train_df['Embarked'] = train_df['Embarked'].map(size_mapping)
# print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))

# correlation matrix 相关系数矩阵
# colormap = plt.cm.viridis
# plt.figure(figsize=(12,12))
# plt.title('Feature correlations', y=1.05, size=15)
# sns.heatmap(train_df.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()



# # missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(12))
# total = test_df.isnull().sum().sort_values(ascending=False)
# percent = (test_df.isnull().sum()/test_df.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(12))
