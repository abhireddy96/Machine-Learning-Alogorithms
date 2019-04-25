# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.loc[:, ['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = dataset.loc[:, ['Profit']].values

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
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(X_train, y_train)
MLR_pred = MLR.predict(X_test)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(random_state=0)
DTR.fit(X_train, y_train)
DTR_pred = DTR.predict(X_test)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=10, random_state=0)
RFR.fit(X_train, y_train)
RFR_pred = RFR.predict(X_test)

# Fitting SVR to the dataset
from sklearn.svm import SVR
SVR = SVR(kernel='rbf')
SVR.fit(X_train, y_train)
SVR_pred = SVR.predict(X_test)

r2_scores = [r2_score(MLR_pred, y_test),
             r2_score(DTR_pred, y_test),
             r2_score(RFR_pred, y_test),
             r2_score(SVR_pred, y_test)]

mean_absolute_errors = [np.mean(np.absolute(MLR_pred - y_test)),
                        np.mean(np.absolute(DTR_pred - y_test)),
                        np.mean(np.absolute(RFR_pred - y_test)),
                        np.mean(np.absolute(SVR_pred - y_test))]

residual_sum_of_squares = [np.mean((MLR_pred - y_test) ** 2),
                           np.mean((DTR_pred - y_test) ** 2),
                           np.mean((RFR_pred - y_test) ** 2),
                           np.mean((SVR_pred - y_test) ** 2)]

df = {'Algorithm': ['Multiple Linear', 'Decision Tree',  'Random Forest', 'Support Vector'],
      'R2 Score': r2_scores, 'MAE': mean_absolute_errors, 'RSS' : residual_sum_of_squares}
evaluation_report = pd.DataFrame(data=df, columns=['Algorithm', 'R2 Score', 'MAE', 'RSS', ], index=None).round(2)
print(evaluation_report)