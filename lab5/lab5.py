import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

# Convert text to word counts
vectorizer = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(data.data)

print("Shape:", X.shape)

# Select one class for multinomial parameter estimation
class_name = "sci.space"

class_index = list(data.target_names).index(class_name)

X_class = X[data.target == class_index]

# Total count of every word
word_counts = np.asarray(X_class.sum(axis=0)).flatten()

total_words = np.sum(word_counts)

# -------------------------
# MLE
# -------------------------

mle = word_counts / total_words

print("\nMLE probabilities:")
print(mle[:10])

# -------------------------
# MAP with Dirichlet prior
# -------------------------

alpha = 1.0

map_estimate = (word_counts + alpha - 1) / \
               (total_words + len(word_counts) * (alpha - 1))

print("\nMAP probabilities with alpha =", alpha)
print(map_estimate[:10])


# MAP with stronger prior
alpha = 2.0

map_estimate_2 = (word_counts + alpha - 1) / \
                 (total_words + len(word_counts) * (alpha - 1))

print("\nMAP probabilities with alpha =", alpha)
print(map_estimate_2[:10])


# Compare the most frequent words
words = vectorizer.get_feature_names_out()

top_indices = np.argsort(word_counts)[-10:][::-1]

print("\nTop 10 words:")

for i in top_indices:
    print(
        words[i],
        "MLE =", mle[i],
        "MAP =", map_estimate[i]
    )