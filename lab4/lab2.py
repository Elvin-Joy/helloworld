import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# --------------------------------
# Load Dataset
# --------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)

# Display information
print("First 5 rows:")
print(X.head())

print("\nDataset Shape:")
print(X.shape)

print("\nColumn Names:")
print(X.columns)

print("\nData Types:")
print(X.dtypes)

print("\nMissing Values:")
print(X.isnull().sum().sum())

print("\nClass Distribution:")
print(y.value_counts())


# --------------------------------
# Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Feature Scaling
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------
# Logistic Regression
# --------------------------------

model = LogisticRegression(max_iter=10000)

model.fit(X_train_scaled, y_train)


# --------------------------------
# Predictions
# --------------------------------

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)


# --------------------------------
# Evaluation
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))


# --------------------------------
# Confusion Matrix Heatmap
# --------------------------------

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# --------------------------------
# ROC Curve
# --------------------------------

y_prob = model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

print("AUC:", auc)

plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# --------------------------------
# Probability Prediction
# --------------------------------

sample = X_test.iloc[0:1]

sample_scaled = scaler.transform(sample)

probability = model.predict_proba(sample_scaled)

print("\nProbability for sample:")
print(probability)

print(
    "Probability of Malignant:",
    probability[0][0]
)


# --------------------------------
# Additional Exercise 1
# Test Size = 30%
# --------------------------------

X_train30, X_test30, y_train30, y_test30 = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

scaler30 = StandardScaler()

X_train30 = scaler30.fit_transform(X_train30)
X_test30 = scaler30.transform(X_test30)

model30 = LogisticRegression(max_iter=10000)

model30.fit(X_train30, y_train30)

pred30 = model30.predict(X_test30)

print("\nAccuracy with 30% test size:")
print(accuracy_score(y_test30, pred30))


# --------------------------------
# Additional Exercise 2
# Without Scaling
# --------------------------------

model_unscaled = LogisticRegression(max_iter=10000)

model_unscaled.fit(X_train, y_train)

pred_unscaled = model_unscaled.predict(X_test)

print("\nAccuracy without scaling:")
print(accuracy_score(y_test, pred_unscaled))


# --------------------------------
# Additional Exercise 3
# Random State = 10
# --------------------------------

X_train10, X_test10, y_train10, y_test10 = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=10,
    stratify=y
)

scaler10 = StandardScaler()

X_train10 = scaler10.fit_transform(X_train10)
X_test10 = scaler10.transform(X_test10)

model10 = LogisticRegression(max_iter=10000)

model10.fit(X_train10, y_train10)

pred10 = model10.predict(X_test10)

print("\nAccuracy with random state 10:")
print(accuracy_score(y_test10, pred10))


# --------------------------------
# Additional Exercise 4
# Five Most Influential Features
# --------------------------------

coefficients = model.coef_[0]

feature_importance = pd.Series(
    coefficients,
    index=X.columns
)

print("\nFive Most Influential Features:")

print(
    feature_importance
    .abs()
    .sort_values(ascending=False)
    .head(5)
)


# --------------------------------
# Additional Exercise 5
# First Five Test Samples
# --------------------------------

first_five = X_test.iloc[:5]

first_five_scaled = scaler.transform(first_five)

predictions = model.predict(first_five_scaled)
probabilities = model.predict_proba(first_five_scaled)

print("\nFirst Five Predictions:")
print(predictions)

print("\nFirst Five Probabilities:")
print(probabilities)


# --------------------------------
# Additional Exercise 6
# Threshold = 0.7
# --------------------------------

probability_1 = model.predict_proba(X_test_scaled)[:, 1]

threshold_predictions = (
    probability_1 >= 0.7
).astype(int)

precision_07 = precision_score(
    y_test,
    threshold_predictions
)

recall_07 = recall_score(
    y_test,
    threshold_predictions
)

print("\nThreshold = 0.7")

print("Precision:", precision_07)
print("Recall:", recall_07)