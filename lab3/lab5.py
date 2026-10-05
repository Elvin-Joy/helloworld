import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = pd.DataFrame({
    "TV": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
    "Radio": [30, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    "Newspaper": [10, 12, 15, 18, 20, 22, 24, 26, 28, 30, 32, 35],
    "Sales": [11, 13, 16, 18, 20, 23, 25, 28, 30, 33, 35, 38]
})

# Features and target
X = data[["TV", "Radio", "Newspaper"]]
Y = data["Sales"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Models
linear = LinearRegression()

ridge = Ridge(alpha=1.0)

lasso = Lasso(alpha=1.0)

# Train models
linear.fit(X_train, Y_train)
ridge.fit(X_train, Y_train)
lasso.fit(X_train, Y_train)

# Predictions
linear_pred = linear.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)

# Print coefficients
print("Linear Regression Coefficients:")
print(linear.coef_)

print("\nRidge Regression Coefficients:")
print(ridge.coef_)

print("\nLasso Regression Coefficients:")
print(lasso.coef_)

# Metrics
print("\nLinear Regression")
print("MSE:", mean_squared_error(Y_test, linear_pred))
print("R2 Score:", r2_score(Y_test, linear_pred))

print("\nRidge Regression")
print("MSE:", mean_squared_error(Y_test, ridge_pred))
print("R2 Score:", r2_score(Y_test, ridge_pred))

print("\nLasso Regression")
print("MSE:", mean_squared_error(Y_test, lasso_pred))
print("R2 Score:", r2_score(Y_test, lasso_pred))