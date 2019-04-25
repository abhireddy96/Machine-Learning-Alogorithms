# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.loc[:, ['Age', 'EstimatedSalary']].values
y = dataset.loc[:, ['Purchased']]
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
KSVM = SVC(kernel='rbf', random_state=0)
KSVM.fit(X_train, y_train)

# Predicting the Test set results
y_pred = KSVM.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
print(jaccard_similarity_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))
print(log_loss((y_pred == 1),(y_test == 1)))

# Visualising the Training set results
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, KSVM.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[:, 0], X_set[:, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('KNN(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
