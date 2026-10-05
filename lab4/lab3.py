import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------
# Load Dataset
# --------------------------------

data = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

# --------------------------------
# Separate X and Y
# --------------------------------

X = data.drop("Outcome", axis=1)

Y = data["Outcome"]


# --------------------------------
# Train-Test Split
# --------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# =================================
# WITHOUT FEATURE SCALING
# =================================

model_unscaled = LogisticRegression(
    max_iter=1000
)

model_unscaled.fit(X_train, Y_train)

Y_pred_unscaled = model_unscaled.predict(X_test)

print("\nWithout Feature Scaling")

print(
    "Accuracy:",
    accuracy_score(Y_test, Y_pred_unscaled)
)

print(
    "Precision:",
    precision_score(Y_test, Y_pred_unscaled)
)

print(
    "Recall:",
    recall_score(Y_test, Y_pred_unscaled)
)

print(
    "F1 Score:",
    f1_score(Y_test, Y_pred_unscaled)
)


# =================================
# WITH FEATURE SCALING
# =================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


model_scaled = LogisticRegression(
    max_iter=1000
)

model_scaled.fit(
    X_train_scaled,
    Y_train
)

Y_pred_scaled = model_scaled.predict(
    X_test_scaled
)

print("\nWith Feature Scaling")

print(
    "Accuracy:",
    accuracy_score(Y_test, Y_pred_scaled)
)

print(
    "Precision:",
    precision_score(Y_test, Y_pred_scaled)
)

print(
    "Recall:",
    recall_score(Y_test, Y_pred_scaled)
)

print(
    "F1 Score:",
    f1_score(Y_test, Y_pred_scaled)
)