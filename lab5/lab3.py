import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix

# Load dataset
data = pd.read_csv("student_exam.csv")

print(data.head())
print("\nShape:")
print(data.shape)

print("\nData Types:")
print(data.dtypes)

print("\nSummary:")
print(data.describe())

# Features and target
X = data[["Study Hours", "Attendance"]]
y = data["Pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLE model - almost no regularization
mle_model = LogisticRegression(C=1e6, max_iter=1000)
mle_model.fit(X_train, y_train)

# MAP model - L2 regularization
map_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)
map_model.fit(X_train, y_train)


def evaluate(model, name):
    y_pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Coefficients:", model.coef_)


evaluate(mle_model, "MLE Logistic Regression")
evaluate(map_model, "MAP Logistic Regression")

# Compare coefficients
print("\nMLE Coefficients:")
print(mle_model.coef_)

print("\nMAP Coefficients:")
print(map_model.coef_)

# Prediction for new student
new_student = [[6, 76]]
new_student = scaler.transform(new_student)

mle_probability = mle_model.predict_proba(new_student)[0][1]
map_probability = map_model.predict_proba(new_student)[0][1]

print("\nNew Student: 6 hours, 76% attendance")

print("MLE Prediction:", mle_model.predict(new_student)[0])
print("MLE Probability:", mle_probability)

print("MAP Prediction:", map_model.predict(new_student)[0])
print("MAP Probability:", map_probability)

# Additional experiments
for C in [0.1, 10]:
    model = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    print("\nMAP with C =", C)
    print("Coefficients:", model.coef_)