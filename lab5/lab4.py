import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())

print("\nClass Distribution:")
print(pd.Series(y).value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# MLE
mle_model = LogisticRegression(
    penalty=None,
    max_iter=5000
)

mle_model.fit(X_train, y_train)


# MAP - L2
l2_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=5000
)

l2_model.fit(X_train, y_train)


# MAP - L1
l1_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=5000
)

l1_model.fit(X_train, y_train)


def evaluate(model, name):
    y_pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Number of non-zero coefficients:",
          np.count_nonzero(model.coef_))


evaluate(mle_model, "MLE")
evaluate(l2_model, "MAP - L2")
evaluate(l1_model, "MAP - L1")


# Compare coefficients
print("\nFirst 10 MLE coefficients:")
print(mle_model.coef_[0][:10])

print("\nFirst 10 L2 coefficients:")
print(l2_model.coef_[0][:10])

print("\nFirst 10 L1 coefficients:")
print(l1_model.coef_[0][:10])