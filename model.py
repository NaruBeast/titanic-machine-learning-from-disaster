import pandas as pd
import matplotlib as plt
import numpy as np

df = pd.read_csv('train.csv')
df.dropna(inplace=True)

df_test = pd.read_csv('test.csv')
df_test.dropna(inplace=True)

X_test = df.iloc[:,[2,4,5,6,7,9]].values

y = df.iloc[:,[1]].values
X = df.iloc[:,[2,4,5,6,7,9]].values

ids = df['PassengerId']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X[:,1] = le_x.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x_test = LabelEncoder()
X_test[:,1] = le_x.fit_transform(X_test[:,1])
onehotencoder_test = OneHotEncoder(categorical_features=[1])
X_test = onehotencoder_test.fit_transform(X_test).toarray()

from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier(n_estimators=5, random_state=0)
rfc.fit(X_train,y_train)