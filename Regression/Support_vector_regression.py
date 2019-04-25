# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.loc[:, ['Level']]
y = dataset.loc[:, ['Salary']]

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
SVR = SVR(kernel='rbf')
SVR.fit(X, y)

# Visualising the SVR results
plt.scatter(X, y, color='red')
plt.plot(X, SVR.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()