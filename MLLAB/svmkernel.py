
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
train_data = pd.read_csv("fashion-mnist_train.csv")
test_data = pd.read_csv("fashion-mnist_test.csv")

# 2. Separate features and labels
X_train_full = train_data.drop("label", axis=1).values
y_train_full = train_data["label"].values

X_test_full = test_data.drop("label", axis=1).values
y_test_full = test_data["label"].values

# 3. Normalize using MinMaxScaler
scaler = MinMaxScaler()

X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

# 4. Select smaller subsets
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=3000,
    random_state=42,
    stratify=y_train_full
)

X_test, _, y_test, _ = train_test_split(
    X_test_full, y_test_full,
    train_size=1000,
    random_state=42,
    stratify=y_test_full
)

# 5. Create SVM models
models = {
    "Linear": SVC(kernel="linear"),
    "Polynomial": SVC(kernel="poly", degree=3),
    "RBF": SVC(kernel="rbf")
}

results = []

# 6. Train and evaluate models
for name, model in models.items():

    print("TRAINING", name, "SVM...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Kernel": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1": f1_score(y_test, predictions, average="weighted", zero_division=0)
    })

# 7. Display comparison
results_df = pd.DataFrame(results).set_index("Kernel")

print("\nSVM Kernel Comparison:")
print(results_df.round(4))

# 8. Plot accuracy comparison
results_df["Accuracy"].plot(
    kind="bar",
    ylim=(0, 1),
    title="SVM Kernel Accuracy Comparison",
    ylabel="Accuracy"
)

plt.show()
