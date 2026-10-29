
# EXPERIMENT 15
# Hyperparameter Tuning on Fashion MNIST
# Compatible with Google Colab and VS Code

import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# --------------------------------------------------
# 2. PREPROCESS THE DATA
# --------------------------------------------------

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("Data preprocessing completed.")

# --------------------------------------------------
# 3. DEFINE HYPERPARAMETER COMBINATIONS
# --------------------------------------------------

settings = [
    (0.001, 32, 5),
    (0.01, 32, 5),
    (0.0001, 32, 5),
    (0.001, 64, 5),
    (0.001, 32, 10)
]

results = []

# --------------------------------------------------
# 4. TRAIN AND EVALUATE EACH MODEL
# --------------------------------------------------

for lr, batch_size, epochs in settings:

    print("\n--------------------------------")
    print("Learning Rate:", lr)
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)
    print("--------------------------------")

    # Create neural network
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model and measure training time
    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=0
    )

    training_time = time.time() - start_time

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    # Save the results
    results.append({
        "Learning Rate": lr,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Test Accuracy (%)": round(test_accuracy * 100, 2),
        "Test Loss": round(test_loss, 4),
        "Training Time (seconds)": round(training_time, 2)
    })

    print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
    print("Test Loss:", round(test_loss, 4))
    print("Training Time:", round(training_time, 2), "seconds")

# --------------------------------------------------
# 5. DISPLAY THE RESULTS TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== FINAL RESULTS ==========")
print(results_df.to_string(index=False))

# Identify the best configuration among those tested
best_index = results_df["Test Accuracy (%)"].idxmax()

print("\n========== BEST CONFIGURATION ==========")
print(results_df.loc[best_index].to_string())

# --------------------------------------------------
# 6. PLOT TEST ACCURACY COMPARISON
# --------------------------------------------------

labels = [
    f"LR={lr}\nBatch={batch}\nEpochs={epochs}"
    for lr, batch, epochs in settings
]

plt.figure(figsize=(12, 5))
plt.bar(labels, results_df["Test Accuracy (%)"])
plt.xlabel("Hyperparameter Configuration")
plt.ylabel("Test Accuracy (%)")
plt.title("Fashion MNIST: Hyperparameter Tuning")
plt.xticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. PLOT TRAINING TIME COMPARISON
# --------------------------------------------------

plt.figure(figsize=(12, 5))
plt.bar(labels, results_df["Training Time (seconds)"])
plt.xlabel("Hyperparameter Configuration")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time for Different Configurations")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. DISCUSS THE RESULTS
# --------------------------------------------------

print("\n========== OBSERVATIONS ==========")
print("1. Learning rate affects the speed and stability of learning.")
print("2. Batch size affects the number of weight updates per epoch.")
print("3. More epochs may improve accuracy but increase training time.")
print("4. The best configuration is selected by test accuracy.")
print("5. Results may vary depending on hardware and random initialization.")
