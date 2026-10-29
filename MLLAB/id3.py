#9
# Part A: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Part B: Download and load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
data = pd.read_excel(url)

# Part C: Clean the data
data = data.dropna(subset=["CustomerID"])
data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]
data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]

data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

# Part D: Create customer features
customers = data.groupby("CustomerID").agg(
    TotalSpend=("TotalPrice", "sum"),
    Purchases=("InvoiceNo", "nunique"),
    Quantity=("Quantity", "sum"),
    Products=("StockCode", "nunique")
).reset_index()

# Create Low, Medium, High spending classes
customers["Segment"] = pd.qcut(
    customers["TotalSpend"].rank(method="first"),
    q=3,
    labels=[0, 1, 2]
)

# Input features and target
X = customers[["Purchases", "Quantity", "Products"]]
y = customers["Segment"].astype(int)

# Part E: Split data and train the tree
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    criterion="entropy", max_depth=4, random_state=42
)
model.fit(X_train, y_train)

# Part F: Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

# Part G: Visualize the decision tree
plt.figure(figsize=(16, 7))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Low", "Medium", "High"],
    filled=True,
    rounded=True
)
plt.show()

# Part H: Feature importance
importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar")
plt.ylabel("Importance")
plt.title("Customer Feature Importance")
plt.show()
