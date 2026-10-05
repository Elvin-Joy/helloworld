import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15
]).reshape(-1, 1)

Y = np.array([
    5, 8, 15, 24, 36, 50, 67, 85,
    108, 130, 158, 190, 225, 265, 310
])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

degrees = [1, 2, 3, 4, 5, 6]

train_errors = []
test_errors = []

print("Degree\tTraining MSE\tTesting MSE")

for degree in degrees:

    # Polynomial features
    poly = PolynomialFeatures(degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Model
    model = LinearRegression()

    model.fit(X_train_poly, Y_train)

    # Predictions
    Y_train_pred = model.predict(X_train_poly)
    Y_test_pred = model.predict(X_test_poly)

    # MSE
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    print(degree, "\t", train_mse, "\t", test_mse)

# Find best degree
best_degree = degrees[np.argmin(test_errors)]

print("\nBest Degree:", best_degree)

# Plot training error
plt.plot(degrees, train_errors, marker="o")

plt.xlabel("Polynomial Degree")
plt.ylabel("Training MSE")
plt.title("Training Error vs Degree")
plt.show()

# Plot testing error
plt.plot(degrees, test_errors, marker="o")

plt.xlabel("Polynomial Degree")
plt.ylabel("Testing MSE")
plt.title("Testing Error vs Degree")
plt.show()

# Plot both errors
plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, test_errors, marker="o", label="Testing Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.show()