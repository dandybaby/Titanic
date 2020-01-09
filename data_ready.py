import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# 导入数据
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
print('训练数据规模：', train.shape)
print('测试数据规模：', test.shape)
full = train.append(test, ignore_index=True)
print(full.head())
print(full.shape)
print(full.info())
print(full.describe())
full['Age'] = full['Age'].fillna(full['Age'].mean())
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
print(full.describe())
print(full['Embarked'].value_counts())
full['Cabin'] = full['Cabin'].fillna('U')
full['Embarked'] = full['Embarked'].fillna('S')
print(full.info())
sex_Dict = {'female': 0, 'male': 1}
full['Sex'] = full['Sex'].map(sex_Dict)
print(full['Sex'].head())
EmbarkedDF = pd.DataFrame()
EmbarkedDF = pd.get_dummies(full['Embarked'], prefix='Embarked')
print(EmbarkedDF.head())
full = pd.concat([full, EmbarkedDF], axis=1)
full.drop('Embarked', axis=1, inplace=True)
print(full.head())
PclassDF = pd.DataFrame()
PclassDF = pd.get_dummies(full['Pclass'], prefix='Pclass')
print(PclassDF.head())
full = pd.concat([full, PclassDF], axis=1)
full.drop('Pclass', axis=1, inplace=True)
print(full.head())
CabinDF = pd.DataFrame()
CabinDF = full['Cabin'].map(lambda a: a[0])
print(CabinDF.head())
CabinDF = pd.get_dummies(CabinDF, prefix='Cabin')
print(CabinDF.head())
full = pd.concat([full, CabinDF], axis=1)
full.drop('Cabin', axis=1, inplace=True)
print(full.head())
FamilyDF = pd.DataFrame()
FamilyDF['familySize'] = full['Parch'] + full['SibSp'] + 1
print(FamilyDF.head())
FamilyDF['familySingle'] = FamilyDF['familySize'].map(lambda b: 1 if b <= 1 else 0)
FamilyDF['familymiddle'] = FamilyDF['familySize'].map(lambda b: 1 if 2 <= b <= 4 else 0)
FamilyDF['familylarge'] = FamilyDF['familySize'].map(lambda b: 1 if b >= 5 else 0)
print(FamilyDF.head())
full = pd.concat([full, FamilyDF], axis=1)
print(full.head())


# 定义函数取所有name的头衔
def firstget(name):
    str4 = name.split(',')[1]
    str5 = str4.split('.')[0]
    str6 = str5.strip()
    return str6


titleDF = pd.DataFrame()
titleDF['Title'] = full['Name'].map(firstget)
print(titleDF.head())
# 根据网上的头衔类别定义如下映射关系
title_mapDict = {'Capt': 'Officer',
                 'Col': 'Officer',
                 'Majior': 'Officer',
                 'Jonkheer': 'Royalty',
                 'Don': 'Royalty',
                 'Sir': 'Royalty',
                 'Dr': 'Officer',
                 'Rev': 'Officer',
                 'the Countess': 'Royalty',
                 'Dona': 'Royalty',
                 'Mme': 'Mrs',
                 'Mlle': 'Miss',
                 'Ms': 'Mrs',
                 'Mr': 'Mr',
                 'Mrs': 'Mrs',
                 'Miss': 'Miss',
                 'Master': 'Master',
                 'Lady': 'Royalty'
                 }
titleDF['Title'] = titleDF['Title'].map(title_mapDict)
titleDF = pd.get_dummies(titleDF)
print(titleDF.head())
full = pd.concat([full, titleDF], axis=1)
print(full.head())
corrDF = full.corr()
print(corrDF)
print(corrDF['Survived'].sort_values(ascending=False))
full_x = pd.concat([titleDF, PclassDF, full['Fare'], CabinDF, EmbarkedDF, FamilyDF, full['Age'], full['Sex']], axis=1)
print(full_x.head())
# 选取样本特征数据
sourceRow = 891
# 选取样本特征数据：从新建的full_x中选取标准化后的前891列为数据特征
source_x = full_x.loc[0:sourceRow - 1, :]
source_y = full.loc[0:sourceRow - 1, 'Survived']
pre_x = full_x.loc[sourceRow:, :]
print('样本特征数据:', source_x.shape)
print('样本标签数据:', source_y.shape)
print('预测特征数据:', pre_x.shape)
from sklearn.model_selection import train_test_split

# 训练数据特征train_x
# 测试数据特征test_x
# 训练数据标签train_y
# 测试数据标签test_y
train_x, test_x, train_y, test_y = train_test_split(source_x, source_y, train_size=0.8)
print('训练数据特征', train_x.shape, '训练数据标签', train_y.shape)
print('测试数据特征', test_x.shape, '测试数据标签', test_y.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')
print(model.fit(train_x, train_y))
print(model.score(test_x,test_y))
pre_y=model.predict(pre_x)
pre_y=pre_y.astype('int')
print(pre_y)
Passenger_ID=full.loc[sourceRow:,'PassengerId']
preDF=pd.DataFrame({'PassengerId':Passenger_ID,
                    'Survived':pre_y})
print(preDF.head())
preDF.to_csv('titanic_predict.csv',index=False)
