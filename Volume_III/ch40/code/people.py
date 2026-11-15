# Invented people: heights (cm) and weights (kg) of three groups.
import numpy as np

# true parameters: weight pi, mean, covariance of each group
GROUPS = [
    (0.30, [132, 31], [[81, 36], [36, 25]]),        # children
    (0.40, [163, 62], [[36, 24], [24, 64]]),        # women
    (0.30, [177, 80], [[49, 7], [7, 81]]),          # men
]


def people_2d(n=500, seed=0):
    """n people; returns X (n x 2) and the hidden group g (n,)."""
    rng = np.random.default_rng(seed)
    pi = [w for w, _, _ in GROUPS]
    g = rng.choice(3, size=n, p=pi)
    X = np.empty((n, 2))
    for k, (_, mu, cov) in enumerate(GROUPS):
        m = g == k
        X[m] = rng.multivariate_normal(mu, cov, size=m.sum())
    return X, g


def heights_1d(n=400, seed=1):
    """Heights of a population of two groups (one feature)."""
    rng = np.random.default_rng(seed)
    g = rng.random(n) < 0.45
    return np.where(g, rng.normal(178, 6.5, n), rng.normal(161, 5.5, n)), g
