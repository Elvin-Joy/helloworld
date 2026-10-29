#13
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# 2. Separate features and target
X = data.drop("quality", axis=1)
y = data["quality"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Define different MLP architectures
models = {
    "32 neurons": (32,),
    "64 neurons": (64,),
    "Two hidden layers (64, 32)": (64, 32)
}

results = []

# 6. Train and evaluate each model
for name, architecture in models.items():
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Architecture": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    })

# 7. Display the results
results_df = pd.DataFrame(results).set_index("Architecture")
print(results_df.round(4))

# 8. Plot accuracy comparison
results_df["Accuracy"].plot(
    kind="bar", ylim=(0, 1),
    title="MLP Architecture Comparison",
    ylabel="Accuracy"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
