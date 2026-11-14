import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

data=pd.read_csv("archive/housing.csv")
print(data.head())
print(data.columns)


X=data["median_income"].values
y=data["median_house_value"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

n=len(X_train)

mean=np.mean(X_train)
std=np.std(X_train)

X_train_scaled=(X_train-mean)/std
X_test_scaled=(X_test-mean)/std


m=0
c=0
lr=0.1

for i in range(1000):
    y_pred=m*X_train_scaled+c
    dm=(-2/n)*np.sum(X_train_scaled*(y_train-y_pred))
    dc=(-2/n)*np.sum(y_train-y_pred)
    m=m-lr*dm
    c=c-lr*dc

gd_pred=m * X_test_scaled + c

print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, gd_pred))
print("R2 Score:", r2_score(y_test, gd_pred))


X_bias=np.c_[np.ones(n),X_train_scaled]
theta=np.linalg.inv(X_bias.T@ X_bias)@X_bias.T@y_train

c2 = theta[0]
m2 = theta[1]

ne_pred = m2 * X_test_scaled + c2

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, ne_pred))
print("R2 Score:", r2_score(y_test, ne_pred))

plt.scatter(X_test, y_test, color="blue", label="Actual Data")
x_line = np.linspace(min(X), max(X), 100)
x_line_scaled = (x_line - mean) / std
plt.plot(x_line, m * x_line_scaled + c,
         color="red", label="Gradient Descent")

plt.plot(x_line, m2 * x_line_scaled + c2,
         color="green", label="Normal Equation")

plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Linear Regression - California Housing")
plt.legend()
plt.show()
