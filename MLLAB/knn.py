
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
train_data = pd.read_csv("fashion-mnist_train.csv")
test_data = pd.read_csv("fashion-mnist_test.csv")

# Separate features and labels
X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

#Normalizing
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use smaller subset to reduce execution time
X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]  

# Different K values
k_values = [1, 3, 5, 7, 9]

accuracies = []
times = []

# Train and test KNN
for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    start = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    end = time.time()

    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy)
    times.append(end - start)

    print("\nK =", k)
    print("Accuracy:", round(accuracy, 4))
    print("Time:", round(end - start, 2), "seconds")

# Plot accuracy
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy Comparison")
plt.grid()
plt.show()

# Plot execution time
plt.plot(k_values, times, marker="o")
plt.xlabel("K Values")
plt.ylabel("Time (seconds)")
plt.title("KNN Execution Time Comparison")
plt.grid()
plt.show()
