
#14

import time
import pandas as pd
import matplotlib.pyplot as plt
import idx2numpy

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load MNIST dataset
X_train = idx2numpy.convert_from_file("archive1/train-images.idx3-ubyte")
y_train = idx2numpy.convert_from_file("archive1/train-labels.idx1-ubyte")

X_test = idx2numpy.convert_from_file("archive1/t10k-images.idx3-ubyte")
y_test = idx2numpy.convert_from_file("archive1/t10k-labels.idx1-ubyte")

# 2. Preprocess images
def preprocess(X):
    X = X.reshape(len(X), 784)
    X = X / 255.0
    return X

X_train = preprocess(X_train)
X_test = preprocess(X_test)

# 3. Define activation functions
activations = ["logistic", "relu", "tanh"]

results = []

# 4. Train and evaluate each network
for act in activations:

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation=act,
        max_iter=20,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    training_time = end - start

    results.append({
        "Activation": act,
        "Accuracy": round(accuracy, 4),
        "Training Time": round(training_time, 2),
        "Iterations": model.n_iter_
    })

    plt.plot(model.loss_curve_, label=act)

# 5. Display comparison
results_df = pd.DataFrame(results)
print("\nActivation Function Comparison:")
print(results_df.to_string(index=False))

# 6. Plot convergence
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.title("Convergence for Different Activation Functions")
plt.legend()
plt.show()

# 7. Plot accuracy comparison
plt.bar(results_df["Activation"], results_df["Accuracy"])
plt.xlabel("Activation Function")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
plt.show()
