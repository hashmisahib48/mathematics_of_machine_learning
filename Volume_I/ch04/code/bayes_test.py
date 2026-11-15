# Bayes' rule for a medical test, exact and by simulation.
import numpy as np

prev, sens, fpr = 0.01, 0.90, 0.05   # P(D), P(+|D), P(+|no D)

p_pos = sens * prev + fpr * (1 - prev)       # Eq. 4.2-2
post = sens * prev / p_pos                   # Eq. 4.2-3
print(f"P(+)   = {p_pos:.4f}")
print(f"P(D|+) = {post:.4f}")

# Second, independent test given the disease status (cond. indep.).
num = sens**2 * prev
post2 = num / (num + fpr**2 * (1 - prev))
print(f"P(D|+,+) = {post2:.4f}")

p2 = sens**2 * prev + fpr**2 * (1 - prev)    # P(+, +)
print(f"P(+,+) = {p2:.6f}   P(+)^2 = {p_pos**2:.6f}")

# Simulate one million people.
rng = np.random.default_rng(0)
n = 1_000_000
sick = rng.random(n) < prev
pos = np.where(sick, rng.random(n) < sens, rng.random(n) < fpr)
print(f"simulated P(+)   = {pos.mean():.4f}")
print(f"simulated P(D|+) = {sick[pos].mean():.4f}")
