import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load California Housing dataset
data = fetch_california_housing()

# Select average number of rooms (AveRooms)
X = data.data[:, [2]]
Y = data.target

# Train-validation split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

degrees = [1, 2, 3, 4, 5, 6]

train_errors = []
validation_errors = []

for degree in degrees:

    # Polynomial features
    poly = PolynomialFeatures(degree)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Model
    model = LinearRegression()

    model.fit(X_train_poly, Y_train)

    # Predictions
    Y_train_pred = model.predict(X_train_poly)
    Y_val_pred = model.predict(X_val_poly)

    # MSE
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    validation_mse = mean_squared_error(Y_val, Y_val_pred)

    train_errors.append(train_mse)
    validation_errors.append(validation_mse)

    print("Degree:", degree)
    print("Training MSE:", train_mse)
    print("Validation MSE:", validation_mse)
    print()

# Plot error curves
plt.plot(degrees, train_errors, marker="o", label="Training MSE")
plt.plot(degrees, validation_errors, marker="o", label="Validation MSE")

plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.show()