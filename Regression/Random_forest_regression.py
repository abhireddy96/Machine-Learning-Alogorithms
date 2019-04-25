# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.loc[:, ['Level']]
y = dataset.loc[:, ['Salary']]

# Fitting Random Forest Regression to the dataset
RFR = RandomForestRegressor(n_estimators=10, random_state=0)
RFR.fit(X, y)

# Visualising the Decision Tree Regression results
plt.scatter(X, y, color='red')
plt.plot(X, RFR.predict(X), color='blue')
plt.title('Truth or Bluff (RFR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()