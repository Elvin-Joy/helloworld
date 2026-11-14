
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"
data = pd.read_csv(url)

# Select columns and remove missing values
data = data[["displacement", "mpg"]].dropna()

X = data[["displacement"]]
y = data["mpg"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (Standardization)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Linear Regression line
X_curve = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_curve_scaled = scaler.transform(X_curve)

y_curve = model.predict(X_curve_scaled)

plt.plot(X_curve, y_curve, label="Linear Regression", color="black")

# Polynomial Regression
for degree in [2, 3, 4]:

    poly = PolynomialFeatures(degree)

    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    print("\nPolynomial Degree:", degree)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Plot polynomial curve
    X_curve_poly = poly.transform(X_curve_scaled)
    y_curve = model.predict(X_curve_poly)

    plt.plot(X_curve, y_curve, label="Degree " + str(degree))

# Plot actual data
plt.scatter(X, y, color="blue", label="Actual Data")

plt.xlabel("Engine Displacement")
plt.ylabel("MPG")
plt.title("Linear and Polynomial Regression")
plt.legend()
plt.show()
 