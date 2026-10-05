import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

data = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1])

n = len(data)
x = np.sum(data)

alpha = 5
beta_value = 2

# MLE
mle = x / n

# MAP
map_estimate = (x + alpha - 1) / (n + alpha + beta_value - 2)

print("Number of Passes =", x)
print("Number of Fails =", n - x)
print("MLE Estimate =", mle)
print("MAP Estimate =", map_estimate)

# Additional priors
map_22 = (x + 2 - 1) / (n + 2 + 2 - 2)
map_101 = (x + 10 - 1) / (n + 10 + 1 - 2)

print("\nMAP with Beta(2,2) =", map_22)
print("MAP with Beta(10,1) =", map_101)

# Plot
theta = np.linspace(0.001, 0.999, 500)

prior = beta.pdf(theta, alpha, beta_value)

# Likelihood
likelihood = theta**x * (1 - theta)**(n - x)
likelihood = likelihood / np.max(likelihood)

# Posterior
posterior = beta.pdf(theta, x + alpha, n - x + beta_value)
posterior = posterior / np.max(posterior)

plt.plot(theta, prior, label="Prior")
plt.plot(theta, likelihood, label="Likelihood")
plt.plot(theta, posterior, label="Posterior")

plt.axvline(mle, linestyle="--", label="MLE")
plt.axvline(map_estimate, linestyle="--", label="MAP")

plt.xlabel("Theta")
plt.ylabel("Density")
plt.title("MLE and MAP Estimation")
plt.legend()
plt.show()