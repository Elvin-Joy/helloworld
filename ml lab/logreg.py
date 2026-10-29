#experiment 6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
url = "https://raw.githubusercontent.com/Zue77/Pima-Indians-Diabetes-Dataset/main/diabetes.csv"
data = pd.read_csv(url)

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Function to evaluate model
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1-score :", f1_score(y_test, y_pred, zero_division=0))

# 1. Without feature scaling
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

print("Without Feature Scaling:")
evaluate(model1, X_test, y_test)

# 2. With feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_scaled, y_train)

print("\nWith Feature Scaling:")
evaluate(model2, X_test_scaled, y_test)
