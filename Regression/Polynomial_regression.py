# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.loc[:, ['Level']]
y = dataset.loc[:, ['Salary']]

# Fitting Polynomial Regression to the dataset
# Transform X into polynomial of degree 4
PR = PolynomialFeatures(degree=4)
X_poly = PR.fit_transform(X)
PR.fit(X_poly, y)
# Fitting Polynomial into Linear Regression to get Polynomial Regression
PLR = LinearRegression()
PLR.fit(X_poly, y)

print("Mean absolute error: %.2f" % np.mean(np.absolute(PLR.predict(PR.fit_transform(X)) - y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((PLR.predict(PR.fit_transform(X)) - y) ** 2))
print("R2-score: %.2f" % r2_score(PLR.predict(PR.fit_transform(X)), y))


# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, PLR.predict(PR.fit_transform(X)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
