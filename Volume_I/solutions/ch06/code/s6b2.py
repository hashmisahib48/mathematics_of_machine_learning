# Problem 6B.2: the winner's curse grows with the number of rules.
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(5)
n_test, n_fresh, reps = 100, 2000, 200


def one_round(k):
    y = rng.integers(0, 2, n_test)                 # coin-flip labels
    X = rng.normal(size=(n_test, k))               # k noise features
    acc = np.mean((X > 0) == y[:, None], axis=0)   # rule j: x_j > 0
    best = acc.max()                           # tune on the test set
    x_new = rng.normal(size=n_fresh)           # fresh data for the
    y_new = rng.integers(0, 2, n_fresh)        # chosen (useless) rule
    return best, np.mean((x_new > 0) == y_new)


def expected_max(k):                               # exact E[max of k]
    a = np.arange(n_test + 1)
    F = binom.cdf(a, n_test, 0.5)
    G = np.r_[0.0, F[:-1]]                         # cdf just below a
    return np.sum(a / n_test * (F ** k - G ** k))


print("rules   best on test   fresh   exact E[best]")
for k in (1, 10, 100, 1000):
    rng = np.random.default_rng(5)             # same seed for each k
    r = np.array([one_round(k) for _ in range(reps)])
    print(f"{k:5d} {r[:, 0].mean():12.3f} {r[:, 1].mean():9.3f}"
          f" {expected_max(k):12.3f}")
