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