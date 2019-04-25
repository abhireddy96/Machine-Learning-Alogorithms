# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.loc[:, ['Country', 'Age', 'Salary']]
y = dataset.loc[:, ['Purchased']]

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', copy=False)
X.loc[:, ['Age', 'Salary']] = imputer.fit_transform(X.loc[:, ['Age', 'Salary']])

# Encoding categorical data
labelencoder_X = LabelEncoder()
X.loc[:, 'Country'] = labelencoder_X.fit_transform(X.loc[:, 'Country'])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y.values.ravel())

# One Hot Encoding data
one_hot_x = pd.get_dummies(X.loc[:, 'Country'])
X = X.drop('Country', axis=1).join(one_hot_x)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
