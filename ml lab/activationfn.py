#14

import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Normalize and flatten the images
X_train = X_train.reshape(len(X_train), 784) / 255.0
X_test = X_test.reshape(len(X_test), 784) / 255.0

# 3. Define activation functions
activations = ["sigmoid", "relu", "tanh"]

results = []
histories = {}

# 4. Train and evaluate each network
for activation in activations:
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    start = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    training_time = time.time() - start

    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    results.append({
        "Activation": activation,
        "Training Time (seconds)": round(training_time, 2),
        "Test Accuracy": round(test_accuracy, 4)
    })

    histories[activation] = history

# 5. Display comparison
results_df = pd.DataFrame(results)
print("\nActivation Function Comparison:")
print(results_df.to_string(index=False))

# 6. Plot training loss to compare convergence
for activation in activations:
    plt.plot(
        histories[activation].history["loss"],
        label=activation
    )

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

# 7. Plot test accuracy comparison
plt.bar(results_df["Activation"], results_df["Test Accuracy"])
plt.title("Test Accuracy Comparison")
plt.xlabel("Activation Function")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.show()
