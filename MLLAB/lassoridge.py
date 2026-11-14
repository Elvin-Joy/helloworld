
# Ridge and Lasso Regression - Diabetes Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes(as_frame=True, scaled=False)

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

# Ridge Regression
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5))
])

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# Lasso Regression
lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCV(alphas=[0.01, 0.1, 1, 10, 100],
                      cv=5, max_iter=10000))
])

lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

# Compare models
models = {
    "Linear": linear_pred,
    "Ridge": ridge_pred,
    "Lasso": lasso_pred
}

mse_values = []
r2_values = []

for name, pred in models.items():
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    mse_values.append(mse)
    r2_values.append(r2)

    print("\n", name, "Regression")
    print("MSE:", round(mse, 2))
    print("R2 Score:", round(r2, 4))

print("\nBest Ridge Alpha:",
      ridge.named_steps["model"].alpha_)

print("Best Lasso Alpha:",
      lasso.named_steps["model"].alpha_)

# MSE and R2 comparison graphs
names = list(models.keys())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(names, mse_values)
plt.title("MSE Comparison")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
plt.bar(names, r2_values)
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.show()

# Actual vs Predicted graph
plt.figure(figsize=(7, 5))

for name, pred in models.items():
    plt.scatter(y_test, pred, label=name)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()
