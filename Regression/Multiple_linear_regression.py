# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.loc[:, ['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = dataset.loc[:, ['Profit']]

# Encoding categorical data
labelencoder_X = LabelEncoder()
X.loc[:, 'State'] = labelencoder_X.fit_transform(X.loc[:, 'State'])

# One Hot Encoding data
one_hot_x = pd.get_dummies(X.loc[:, 'State'])
X = X.drop('State', axis=1).join(one_hot_x)

# Avoiding the Dummy Variable Trap
X = X.iloc[:, :-1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Simple Linear Regression to the Training set
MLR = LinearRegression()
MLR.fit(X_train, y_train)

# Predicting the Test set results
y_pred = MLR.predict(X_test)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test)))
print("Residual sum of squares: %.2f"% np.mean((y_pred - y_test) ** 2))
print('Variance score: %.2f' % MLR.score(X_test, y_test))
print("R2-score: %.2f" % r2_score(y_pred, y_test))