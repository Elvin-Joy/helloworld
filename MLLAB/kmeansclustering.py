
# K-means Clustering on Digits Dataset

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Load Digits dataset
digits = load_digits()
print(digits.feature_names)
X = digits.data
y = digits.target

# 2. Preprocess the data
def preprocess(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

X = preprocess(X)

# 3. Try different numbers of clusters
k_values = [2, 5, 10, 15, 20]

results = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    inertia = model.inertia_
    silhouette = silhouette_score(X, model.labels_)

    results.append({
        "K": k,
        "Inertia": round(inertia, 2),
        "Silhouette Score": round(silhouette, 4)
    })

# 4. Display results
results_df = pd.DataFrame(results)

print("\nK-means Clustering Results:")
print(results_df.to_string(index=False))

# 5. Plot Inertia
plt.plot(results_df["K"], results_df["Inertia"], marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.show()

# 6. Plot Silhouette Score
plt.plot(results_df["K"], results_df["Silhouette Score"], marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.grid()
plt.show()
