
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MLE - No Regularization
mle = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty=None, max_iter=10000))
])

# MAP - L2 Regularization
l2 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(penalty="l2", C=1.0, max_iter=10000))
])

# MAP - L1 Regularization
l1 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l1", C=1.0, solver="liblinear", max_iter=10000
    ))
])

# Store models
models = {
    "MLE": mle,
    "MAP - L2": l2,
    "MAP - L1": l1
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    print("\n", name)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Log Loss:", round(log_loss(y_test, prob), 4))

# Compare coefficients
coefficients = pd.DataFrame({
    name: model.named_steps["model"].coef_[0]
    for name, model in models.items()
}, index=data.feature_names)

print("\nCoefficient Comparison (First 10 Features):")
print(coefficients.head(10).round(4))

# Count non-zero coefficients
for name, model in models.items():
    coef = model.named_steps["model"].coef_[0]
    print(name, "- Non-zero coefficients:", np.count_nonzero(coef))

# Accuracy comparison
names = list(models.keys())

accuracies = [
    accuracy_score(y_test, model.predict(X_test))
    for model in models.values()
]

plt.bar(names, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("MLE vs MAP Logistic Regression")
plt.show()

# Coefficient comparison
coefficients.head(10).plot(kind="bar", figsize=(10, 5))
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Parameter Estimates: MLE vs MAP")
plt.tight_layout()
plt.show()
