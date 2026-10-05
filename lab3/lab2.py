import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"

data = pd.read_csv(url)

# Remove missing values
data = data.dropna()

# X = engine displacement
# Y = MPG
X = data[["displ"]]
Y = data["mpg"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("linear", LinearRegression())
])

# Polynomial degrees
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

# Test prediction
Y_pred = grid.predict(X_test)

# Metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Test MSE:", mse)
print("Test R2 Score:", r2)

# Plot
X_curve = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
Y_curve = grid.predict(X_curve)

plt.scatter(X, Y, label="Data Points")
plt.plot(X_curve, Y_curve, label="Best Polynomial Curve")

plt.xlabel("Engine Displacement")
plt.ylabel("MPG")
plt.title("Auto MPG Polynomial Regression")
plt.legend()
plt.show()
