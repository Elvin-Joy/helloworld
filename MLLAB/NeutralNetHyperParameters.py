
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
train_data = pd.read_csv("fashion-mnist_train.csv")
test_data = pd.read_csv("fashion-mnist_test.csv")

# 2. Separate features and target
X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

# 3. Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Define hyperparameter combinations
experiments = [
    {"name": "Baseline", "lr": 0.001, "batch": 128, "epochs": 5},
    {"name": "High LR", "lr": 0.01, "batch": 128, "epochs": 5},
    {"name": "Low LR", "lr": 0.0001, "batch": 128, "epochs": 5},
    {"name": "Small Batch", "lr": 0.001, "batch": 32, "epochs": 5},
    {"name": "Large Batch", "lr": 0.001, "batch": 256, "epochs": 5},
    {"name": "More Epochs", "lr": 0.001, "batch": 128, "epochs": 10}
]

results = []

# 5. Train and evaluate each model
for exp in experiments:

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        learning_rate_init=exp["lr"],
        batch_size=exp["batch"],
        max_iter=exp["epochs"],
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append({
        "Experiment": exp["name"],
        "Learning Rate": exp["lr"],
        "Batch Size": exp["batch"],
        "Epochs": exp["epochs"],
        "Accuracy": accuracy_score(y_test, predictions)
    })

# 6. Display results
results_df = pd.DataFrame(results).set_index("Experiment")
print(results_df.round(4))

# 7. Plot accuracy comparison
results_df["Accuracy"].plot(
    kind="bar",
    ylim=(0, 1),
    title="MLP Hyperparameter Comparison",
    ylabel="Accuracy"
)


plt.tight_layout()
plt.show()
