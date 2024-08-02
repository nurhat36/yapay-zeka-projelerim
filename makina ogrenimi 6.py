import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
##ODEV

# Load the dataset
dataframe = pd.read_excel("iris.xlsx")
print(dataframe.head())

# Visualize the data
sbn.scatterplot(data=dataframe)
plt.show()

# Selecting features and target variables
x = dataframe.iloc[:, 1:4].values
y = dataframe.iloc[:, 4].values  # Assuming y is a single column for logistic regression

print("x shape:", x.shape)
print("y shape:", y.shape)


# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Standardizing the data
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the Logistic Regression model
log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train, y_train)

# Making predictions
y_pred = log_reg.predict(x_test)
print("Predictions:", y_pred)


cm=confusion_matrix(y_test,y_pred)
print(cm)
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

svc=SVC(kernel="rbf")
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)


gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print("gnb")
print(cm)

dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print("dtc")
print(cm)


rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print("rfc")
print(cm)
