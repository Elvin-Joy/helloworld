import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
Y = np.array([30, 35, 40, 45, 50, 56, 61, 66], dtype=float)

# Initial values
b0 = 0
b1 = 0
# Gradient Descent settings
learning_rate = 0.01
iterations = 1000
# Store cost values
cost_history = []

# Number of data points
n = len(X)
for i in range(iterations):

    # Predictions
    Y_pred = b0 + b1 * X

    # Error
    error = Y_pred - Y

    # Calculate gradients
    gradient_b0 = (1 / n) * np.sum(error)
    gradient_b1 = (1 / n) * np.sum(error * X)

    # Update parameters
    b0 = b0 - learning_rate * gradient_b0
    b1 = b1 - learning_rate * gradient_b1

    # Calculate cost
    cost = (1 / (2 * n)) * np.sum(error ** 2)

    # Store cost
    cost_history.append(cost)

# Print final coefficients
print("Intercept (b0):", b0)
print("Slope (b1):", b1)
Y_pred = b0 + b1 * X
# -------------------------------
# Graph 1: Data + Regression Line
# -------------------------------

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")

plt.xlabel("Experience (Years)")
plt.ylabel("Salary (₹ Thousand)")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# -------------------------------
# Graph 2: Cost vs Iterations
# -------------------------------

plt.plot(range(iterations), cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost vs Iterations")

plt.show()  