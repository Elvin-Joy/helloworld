# HYPERPARAMETER TUNING
# Fashion MNIST Dataset

import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. LOAD DATASET
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

# 2. NORMALIZE PIXEL VALUES
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. FLATTEN IMAGES
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 4. USE SMALL SUBSET
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

# 5. HYPERPARAMETERS
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [10, 20]

# 6. TRY DIFFERENT COMBINATIONS
for lr in learning_rates:
    for batch_size in batch_sizes:
        for epoch in epochs:
            print("\nLearning Rate:", lr)
            print("Batch Size:", batch_size)
            print("Epochs:", epoch)
            model=sequential(
                Input(shape=(784,)),
                Dense(100,activation="relu"),
                Dense(10,activation="softmax")
            )

            # 7. CREATE MODEL
            model = Sequential([
                Input(shape=(784,)),
                Dense(100, activation="relu"),
                Dense(10, activation="softmax")
            ])
            model.compile(
                optimizer=Adam(learning_rate=lr),
                metrics=["accuracy"],
                loss="Sparse_categorical_crossentropy"
            )

            # 8. COMPILE MODEL
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            # 9. TRAIN MODEL
            start = time.time()
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=0)
            end = time.time()

            # 10. PREDICT
            y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

            # 11. EVALUATE
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            print("Accuracy :", accuracy)
            print("Precision:", precision)
            print("Recall   :", recall)
            print("F1 Score :", f1)
            print("Training Time:", end - start, "seconds")