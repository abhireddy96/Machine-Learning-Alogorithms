import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.loc[:, ['Age', 'EstimatedSalary']]
y = dataset.loc[:, ['Purchased']]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression(random_state=0)
LRC.fit(X_train, y_train)
LRC_pred = LRC.predict(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
SVM = SVC(kernel = 'linear', random_state = 0)
SVM.fit(X_train, y_train)
SVM_pred = SVM.predict(X_test)

# Fitting Kernel SVM to the Training set
KSVM = SVC(kernel='rbf', random_state=0)
KSVM.fit(X_train, y_train)
KSVM_pred = KSVM.predict(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
KNN.fit(X_train, y_train)
KNN_pred = KNN.predict(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)
NB_pred = NB.predict(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
DT.fit(X_train, y_train)
DT_pred = DT.predict(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
RF.fit(X_train, y_train)
RF_pred = RF.predict(X_test)

jaccard_score = [jaccard_similarity_score(y_test, KNN_pred),
                 jaccard_similarity_score(y_test, NB_pred),
                 jaccard_similarity_score(y_test, SVM_pred),
                 jaccard_similarity_score(y_test, LRC_pred),
                 jaccard_similarity_score(y_test, DT_pred),
                 jaccard_similarity_score(y_test, KSVM_pred),
                 jaccard_similarity_score(y_test, RF_pred)]

F1_score = [f1_score(y_test, KNN_pred, average='weighted'),
            f1_score(y_test, NB_pred, average='weighted'),
            f1_score(y_test, SVM_pred, average='weighted'),
            f1_score(y_test, LRC_pred, average='weighted'),
            f1_score(y_test, DT_pred, average='weighted'),
            f1_score(y_test, KSVM_pred, average='weighted'),
            f1_score(y_test, RF_pred, average='weighted')]


df = {'Algorithm': ['KNN', 'Naive Bayes',  'SVM', 'Logistic Regression', 'Decision Tree', 'Kernel SVM', 'Random Forest'],
      'Jaccard': jaccard_score, 'F1-score': F1_score}
evaluation_report = pd.DataFrame(data=df, columns=['Algorithm', 'Jaccard', 'F1-score'], index=None)
print(evaluation_report)