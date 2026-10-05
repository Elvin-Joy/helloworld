import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load Auto MPG Dataset
# --------------------------------------------------

url = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"

data = pd.read_csv(url)


# --------------------------------------------------
# 2. Preprocess the Dataset
# --------------------------------------------------

# Convert horsepower to numeric
data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")

# Remove rows with missing values
data = data.dropna()

# Select input and output
X = data[["displacement"]]
Y = data["mpg"]


# --------------------------------------------------
# 3. Compare Different Polynomial Degrees
# --------------------------------------------------

degrees = [1, 2, 3, 4]

results = {}

models = {}

for degree in degrees:

    # Create polynomial features
    polynomial = PolynomialFeatures(degree=degree)

    X_poly = polynomial.fit_transform(X)

    # Create linear regression model
    model = LinearRegression()

    # Train model
    model.fit(X_poly, Y)

    # Predict
    Y_pred = model.predict(X_poly)

    # Calculate metrics
    mse = mean_squared_error(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)

    # Store results
    results[degree] = [mse, r2]

    models[degree] = (polynomial, model)


# --------------------------------------------------
# 4. Display Results
# --------------------------------------------------

print("Degree\tMSE\t\tR-squared")

for degree in degrees:

    mse = results[degree][0]
    r2 = results[degree][1]

    print(f"{degree}\t{mse:.4f}\t\t{r2:.4f}")


# --------------------------------------------------
# 5. Visualization
# --------------------------------------------------

# Create smooth X values for plotting
X_plot = np.linspace(
    X["displacement"].min(),
    X["displacement"].max(),
    300
).reshape(-1, 1)


plt.scatter(
    X,
    Y,
    label="Actual Data",
    s=15
)


# Plot each polynomial model
for degree in degrees:

    polynomial, model = models[degree]

    X_plot_poly = polynomial.transform(X_plot)

    Y_plot = model.predict(X_plot_poly)

    plt.plot(
        X_plot,
        Y_plot,
        label=f"Degree {degree}"
    )


plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Polynomial Regression on Auto MPG Dataset")

plt.legend()
plt.show()