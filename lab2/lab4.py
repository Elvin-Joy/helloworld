import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load California Housing Dataset
# --------------------------------------------------

data = fetch_california_housing()

# AveRooms is the 3rd feature (index 2)
X = data.data[:, 2]

# Median house value
Y = data.target

# --------------------------------------------------
# 2. Preprocess X using Standardization
# --------------------------------------------------

X_mean = np.mean(X)
X_std = np.std(X)

X_scaled = (X - X_mean) / X_std

# Number of data points
n = len(X_scaled)

# --------------------------------------------------
# 3. Linear Regression using Normal Equation
# --------------------------------------------------

X_matrix = np.column_stack((np.ones(n), X_scaled))

theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y

b0_normal = theta[0]
b1_normal = theta[1]

Y_pred_normal = b0_normal + b1_normal * X_scaled

# --------------------------------------------------
# 4. Linear Regression using Gradient Descent
# --------------------------------------------------

b0 = 0
b1 = 0

learning_rate = 0.01
iterations = 1000

cost_history = []

for i in range(iterations):

    # Prediction
    Y_pred = b0 + b1 * X_scaled

    # Error
    error = Y_pred - Y

    # Gradients
    gradient_b0 = (1 / n) * np.sum(error)
    gradient_b1 = (1 / n) * np.sum(error * X_scaled)

    # Update parameters
    b0 = b0 - learning_rate * gradient_b0
    b1 = b1 - learning_rate * gradient_b1

    # Cost
    cost = (1 / (2 * n)) * np.sum(error ** 2)

    cost_history.append(cost)

Y_pred_gd = b0 + b1 * X_scaled

# --------------------------------------------------
# 5. Calculate MSE and R-squared
# --------------------------------------------------

mse_normal = mean_squared_error(Y, Y_pred_normal)
r2_normal = r2_score(Y, Y_pred_normal)

mse_gd = mean_squared_error(Y, Y_pred_gd)
r2_gd = r2_score(Y, Y_pred_gd)

# --------------------------------------------------
# 6. Display Results
# --------------------------------------------------

print("----- Normal Equation -----")
print("Intercept:", b0_normal)
print("Slope:", b1_normal)
print("MSE:", mse_normal)
print("R-squared:", r2_normal)

print()

print("----- Gradient Descent -----")
print("Intercept:", b0)
print("Slope:", b1)
print("MSE:", mse_gd)
print("R-squared:", r2_gd)

# --------------------------------------------------
# 7. Visualize Regression Line
# --------------------------------------------------

# Sort X so that the regression line is drawn properly
sorted_indices = np.argsort(X)

X_sorted = X[sorted_indices]
X_scaled_sorted = X_scaled[sorted_indices]
Y_sorted = Y[sorted_indices]

Y_line = b0_normal + b1_normal * X_scaled_sorted

plt.scatter(X, Y, s=10, label="Actual Data")
plt.plot(X_sorted, Y_line, label="Regression Line")

plt.xlabel("Average Number of Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.title("Linear Regression on California Housing Dataset")

plt.legend()
plt.show()

# --------------------------------------------------
# 8. Cost vs Iterations
# --------------------------------------------------

plt.plot(range(iterations), cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost vs Iterations")

plt.show()