import numpy as np

# Dataset
X1 = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
X2 = np.array([10, 8, 6, 5, 3, 2])
Y = np.array([20, 24, 29, 32, 38, 43])

# Create the X matrix
X_matrix = np.column_stack((np.ones(len(X1)), X1, X2))

# Normal Equation
theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y

# Regression coefficients
b0 = theta[0]
b1 = theta[1]
b2 = theta[2]

print("Intercept (b0):", b0)
print("Coefficient for Area (b1):", b1)
print("Coefficient for Age (b2):", b2)

# Prediction for Area = 4 and Age = 1
area = 4
age = 1

predicted_price = b0 + b1 * area + b2 * age

print("Predicted price:", predicted_price, "lakhs")