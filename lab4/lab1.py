import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([
    1.0, 1.5, 2.0, 2.5, 3.0,
    3.5, 4.0, 4.5, 5.0, 5.5,
    6.0, 6.5, 7.0, 7.5, 8.0,
    8.5, 9.0, 9.5, 10.0, 10.5
])

Y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
w = 0
b = 0

learning_rate = 0.1
iterations = 10000

n = len(X)

# Gradient Descent
for i in range(iterations):

    z = w * X + b

    predictions = sigmoid(z)

    error = predictions - Y

    dw = (1 / n) * np.sum(error * X)
    db = (1 / n) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Weight:", w)
print("Bias:", b)

# Predictions
probabilities = sigmoid(w * X + b)

predictions = (probabilities >= 0.5).astype(int)

# Accuracy
accuracy = np.mean(predictions == Y)

print("Accuracy:", accuracy)

# Predict new students
new_hours = np.array([4.5, 6, 9])

new_probabilities = sigmoid(w * new_hours + b)
new_predictions = (new_probabilities >= 0.5).astype(int)

for hours, probability, prediction in zip(
    new_hours, new_probabilities, new_predictions
):
    print(
        "Hours:", hours,
        "Probability:", probability,
        "Prediction:", prediction
    )

# Logistic curve
X_curve = np.linspace(1, 10.5, 200)
Y_curve = sigmoid(w * X_curve + b)

# Plot data
plt.scatter(X, Y, label="Data Points")

plt.plot(
    X_curve,
    Y_curve,
    label="Logistic Regression Curve"
)

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression")
plt.legend()
plt.show()