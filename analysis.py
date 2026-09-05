import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# --------------------------------------------------
# 1. Load observed data
# --------------------------------------------------

df = pd.read_csv("data/observations.csv")

x = df["attempts_until_success"].to_numpy()

n = len(x)
total_attempts = x.sum()

# --------------------------------------------------
# 2. Estimate p using Maximum Likelihood Estimation
# --------------------------------------------------

p_hat = n / total_attempts

print(f"Number of observed successes: {n}")
print(f"Total attempts: {total_attempts}")
print(f"Mean attempts until success: {x.mean():.3f}")
print(f"MLE of success probability p: {p_hat:.4f}")
print(f"Theoretical p under $6 break-even assumption: {1/6:.4f}")

# --------------------------------------------------
# 3. Compare observed data with fitted geometric model
# --------------------------------------------------

max_attempt = x.max()
attempts = np.arange(1, max_attempt + 1)

# Empirical probability for each number of attempts
observed_counts = (
    pd.Series(x)
    .value_counts()
    .reindex(attempts, fill_value=0)
    .sort_index()
)

observed_prob = observed_counts / n

# Probability predicted by fitted geometric distribution
fitted_prob = geom.pmf(attempts, p_hat)

# --------------------------------------------------
# 4. Plot observed vs fitted distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    attempts,
    observed_prob,
    alpha=0.6,
    label="Observed"
)

plt.plot(
    attempts,
    fitted_prob,
    marker="o",
    label=f"Fitted geometric (p={p_hat:.3f})"
)

plt.xlabel("Attempts until first success")
plt.ylabel("Probability")
plt.title("Observed vs Fitted Geometric Distribution")
plt.legend()
plt.tight_layout()

plt.savefig("figures/attempts_distribution.png", dpi=300)
plt.show()
