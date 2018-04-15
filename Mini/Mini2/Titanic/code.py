import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


data = pd.read_csv('train.csv')
data.head()

Y = data.filter(['Survived'], axis=1)
Y['Survived'] = pd.Categorical(Y['Survived'])
Y = np.array(Y['Survived'])
X = data.drop(['PassengerId','Survived', 'Ticket'],axis=1)



categorical = X.dtypes[X.dtypes == "object"].index
print(categorical)

print X[categorical].describe()

## Extract the titles from names (Mr., Mrs., Miss., etc.)
Title = X.Name.str.split(',', expand=True)
Title = Title[1].str.split('.', expand=True)
Title = Title[0]
Title = Title.replace([' the Countess', ' Dr', ' Rev',' Major',' Col', ' Mlle', ' Jonkheer', ' Ms', ' Sir', ' Don', ' Mme', ' Capt', ' Lady'],['Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other'])
Title = pd.Categorical(Title)
X['Name'] = Title

## Extract the cabin section from cabin number
char_cabin = X["Cabin"].astype(str) # Convert data to str
new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter
new_Cabin[new_Cabin=='T'] = 'G'
new_Cabin = pd.Categorical(new_Cabin)
print new_Cabin .describe()
X['Cabin'] = new_Cabin

## Make Embarked Categorical

Embarked = X.Embarked
Embarked = Embarked.fillna(Embarked.value_counts().index[0])
Embarked = pd.Categorical(Embarked)
X['Embarked'] = Embarked

## Make Sex Categorical

Sex = X.Sex
Sex = pd.Categorical(Sex)
X['Sex'] = Sex

## Add dummy variable to indicate missing age value

X['Age_Missing'] = 0
Miss = np.array(X.Age_Missing)
Miss[np.argwhere(np.isnan(X.Age))] = 1
Miss = pd.Categorical(Miss)
X['Age_Missing'] = Miss
Age = X.Age
Age = Age.fillna(0)
Age = pd.DataFrame(Age)
X['Age'] = Age

plt.figure()
X["Fare"].hist()
plt.show()

##X["Fare"].apply(np.log).hist()


XX = X.to_dict('records')
YY = Y.to_dict('records')
V = DictVectorizer(sparse=False)

XXX = V.fit_transform(XX)


pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    RandomForestClassifier(n_estimators = 100, max_depth=6)
)

cv = cross_validate(pipeline, XX, Y,cv=10)
print cv['test_score'].mean()
