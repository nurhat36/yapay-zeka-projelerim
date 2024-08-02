import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


dataframe = pd.read_csv("maaslar.csv")
print(dataframe)
sbn.scatterplot(data=dataframe)
plt.show()

x=dataframe.iloc[:,1:2]
y=dataframe.iloc[:,2:]
X=x.values
Y=y.values



lin_reg=LinearRegression()
lin_reg.fit(X,y)
plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color="blue")
plt.show()

poly_reg=PolynomialFeatures(degree=2)

x_poly=poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()


poly_reg=PolynomialFeatures(degree=4)

x_poly=poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show()

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))   

##support vector regression

sc=StandardScaler()
x_olcekli=sc.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)

svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)
plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
z=X+0.5
k=X-0.5
plt.scatter(X,Y)
plt.plot(X,r_dt.predict(X))
plt.plot(X,r_dt.predict(z),color="blue")
plt.plot(X,r_dt.predict(k),color="red")
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

rfr_reg=RandomForestRegressor(n_estimators=10,random_state=0)

rfr_reg.fit(X,Y.ravel())
print(rfr_reg.predict([[6.6]]))
plt.scatter(X,Y,color="red")
plt.plot(X,rfr_reg.predict(X),color="blue")
plt.plot(X,rfr_reg.predict(z),color="green")
plt.plot(X,r_dt.predict(k),color="yellow")
plt.show()
print("random forest r2 degeri")
print(r2_score(Y,rfr_reg.predict(X)))