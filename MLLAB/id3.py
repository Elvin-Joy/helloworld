# ------------------------------------------------------------
# QUESTION 9: Decision Tree - Online Retail
# Expected CSV: OnlineRetail.csv
# Required columns: CustomerID, InvoiceNo, Quantity, UnitPrice.
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_excel("Online Retail.xlsx")
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

df["CustomerID"] = df["CustomerID"].astype(int)
df["Amount"] = df["Quantity"] * df["UnitPrice"]

customer = df.groupby("CustomerID").agg(
    TotalAmount=("Amount", "sum"),
    Frequency=("InvoiceNo", "nunique"),
    AvgQuantity=("Quantity", "mean"),
    AvgPrice=("Amount", "mean")
)

customer["segment"] = pd.qcut(
    customer["TotalAmount"].rank(method="first"),
    q=3, labels=["low", "medium", "high"]
)

x = customer[["Frequency", "AvgQuantity", "AvgPrice"]]
y = customer["segment"]

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

print("Accuracy Score:", accuracy_score(ytest, pred))
print(classification_report(ytest, pred, zero_division=0))

plt.figure(figsize=(24, 12))
plot_tree(model, feature_names=x.columns,
          class_names=[str(c) for c in model.classes_],
          filled=True, rounded=True)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

importance = pd.Series(
    model.feature_importances_, index=x.columns
).sort_values(ascending=False)
print("\nFeature Importances\n")
print(importance)

