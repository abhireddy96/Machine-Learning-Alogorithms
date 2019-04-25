# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.loc[:, ['Level']]
y = dataset.loc[:, ['Salary']]

# Fitting Decision Tree Regression to the dataset
DTR = DecisionTreeRegressor(random_state=0)
DTR.fit(X, y)

# Visualising the Decision Tree Regression results
plt.scatter(X, y, color='red')
plt.plot(X, DTR.predict(X), color='blue')
plt.title('Truth or Bluff (DTR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()