import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)

Y = np.array([
    3.2, 5.1, 8.8, 15.6, 25.4, 38.2,
    54.1, 73.5, 96.8, 123.2, 154.6, 190.8
])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("linear", LinearRegression())
])

# Degrees 1 to 5
parameters = {
    "poly__degree": [1, 2, 3, 4, 5]
}

# GridSearchCV
grid = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, Y_train)

# Best degree
print("Best Polynomial Degree:", grid.best_params_["poly__degree"])
print("Best Cross-Validation Score:", grid.best_score_)

# Predict test data
Y_pred = grid.predict(X_test)

# Metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Test MSE:", mse)
print("Test R2 Score:", r2)

# Plot
X_curve = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
Y_curve = grid.predict(X_curve)

plt.scatter(X, Y, label="Original Data")
plt.plot(X_curve, Y_curve, label="Best Polynomial Curve")

plt.xlabel("Hours of Practice")
plt.ylabel("Performance Score")
plt.title("Polynomial Regression with GridSearchCV")
plt.legend()
plt.show()