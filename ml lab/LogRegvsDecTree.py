#10

# Experiment 1: Logistic Regression vs Decision Tree
# Dataset: Adult Income

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Part 1: Load the dataset
data = fetch_openml(name="adult", version=2, as_frame=True)

X = data.data
y = data.target

# Part 2: Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns

# Part 3: Preprocess the data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# Part 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Part 5: Create the two models
logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

tree_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ))
])

# Part 6: Train and evaluate both models
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, pos_label=">50K"
        ),
        "Recall": recall_score(
            y_test, predictions, pos_label=">50K"
        ),
        "F1-score": f1_score(
            y_test, predictions, pos_label=">50K"
        )
    })

# Part 7: Display comparison
results_df = pd.DataFrame(results).set_index("Model")

print("Model Performance Comparison:")
print(results_df.round(4))

# Part 8: Plot the F1-score comparison
results_df["F1-score"].plot(
    kind="bar", title="F1-score Comparison",
    ylabel="F1-score", ylim=(0, 1)
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
