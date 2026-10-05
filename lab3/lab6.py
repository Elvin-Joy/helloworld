import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes()

X = data.data
Y = data.target

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --------------------------------
# Linear Regression
# --------------------------------

linear = LinearRegression()

linear.fit(X_train, Y_train)

linear_pred = linear.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(Y_test, linear_pred))
print("R2 Score:", r2_score(Y_test, linear_pred))


# --------------------------------
# Ridge Regression + CV
# --------------------------------

ridge = Ridge()

ridge_parameters = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

ridge_grid = GridSearchCV(
    ridge,
    ridge_parameters,
    cv=5,
    scoring="neg_mean_squared_error"
)

ridge_grid.fit(X_train, Y_train)

ridge_pred = ridge_grid.predict(X_test)

print("\nRidge Regression")
print("Best Alpha:", ridge_grid.best_params_["alpha"])
print("MSE:", mean_squared_error(Y_test, ridge_pred))
print("R2 Score:", r2_score(Y_test, ridge_pred))


# --------------------------------
# Lasso Regression + CV
# --------------------------------

lasso = Lasso(max_iter=10000)

lasso_parameters = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

lasso_grid = GridSearchCV(
    lasso,
    lasso_parameters,
    cv=5,
    scoring="neg_mean_squared_error"
)

lasso_grid.fit(X_train, Y_train)

lasso_pred = lasso_grid.predict(X_test)

print("\nLasso Regression")
print("Best Alpha:", lasso_grid.best_params_["alpha"])
print("MSE:", mean_squared_error(Y_test, lasso_pred))
print("R2 Score:", r2_score(Y_test, lasso_pred))