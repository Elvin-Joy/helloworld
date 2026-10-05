import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data = np.array([52, 49, 51, 53, 48, 50, 54, 51, 49, 52])

n = len(data)

sigma2 = 4
mu0 = 50
sigma02 = 9

# Sample mean
sample_mean = np.mean(data)

# MLE
mle = sample_mean

# MAP
map_estimate = ((n / sigma2) * sample_mean + (1 / sigma02) * mu0) / \
               ((n / sigma2) + (1 / sigma02))

print("Sample Mean =", sample_mean)
print("MLE Estimate =", mle)
print("MAP Estimate =", map_estimate)

# Additional experiments
def calculate_map(prior_mean, prior_variance):
    return ((n / sigma2) * sample_mean + (1 / prior_variance) * prior_mean) / \
           ((n / sigma2) + (1 / prior_variance))

print("\nPrior Mean = 48:")
print(calculate_map(48, 9))

print("\nPrior Variance = 1:")
print(calculate_map(50, 1))

print("\nPrior Variance = 25:")
print(calculate_map(50, 25))

# Plot
mu = np.linspace(47, 55, 500)

prior = norm.pdf(mu, mu0, np.sqrt(sigma02))

likelihood = norm.pdf(mu, sample_mean, np.sqrt(sigma2 / n))

posterior_variance = 1 / ((n / sigma2) + (1 / sigma02))
posterior = norm.pdf(mu, map_estimate, np.sqrt(posterior_variance))

plt.plot(mu, prior, label="Prior")
plt.plot(mu, likelihood, label="Likelihood")
plt.plot(mu, posterior, label="Posterior")

plt.axvline(mle, linestyle="--", label="MLE")
plt.axvline(map_estimate, linestyle="--", label="MAP")

plt.xlabel("Mean")
plt.ylabel("Density")
plt.title("Gaussian MLE and MAP")
plt.legend()
plt.show()