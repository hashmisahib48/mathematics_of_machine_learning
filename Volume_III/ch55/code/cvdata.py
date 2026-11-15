# Invented data sets of Chapter 55 (shared by the scripts).
import numpy as np


def make_patients(n_pat=40, m=8, p=6, seed=0):
    """Rows = visits. Each patient has a fingerprint u_g (all features)
    and a class; only feature 0 carries a weak class signal."""
    rng = np.random.RandomState(seed)
    cls = rng.randint(0, 2, n_pat)               # one class per patient
    u = rng.normal(0, 1.0, (n_pat, p))           # patient fingerprint
    u[:, 0] += 0.6 * (2 * cls - 1)               # weak real signal
    g = np.repeat(np.arange(n_pat), m)
    X = u[g] + rng.normal(0, 0.35, (n_pat * m, p))
    return X, cls[g], g


def make_sales(n=300, seed=0):
    """Weekly sales: trend + season + slow random wander + noise.
    Features known in advance: week number and the season phase."""
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    wander = np.zeros(n)
    for k in range(1, n):
        wander[k] = 0.95 * wander[k - 1] + rng.normal(0, 1.0)
    y = 50 + 0.15 * t + 8 * np.sin(2 * np.pi * t / 52) + 3 * wander \
        + rng.normal(0, 1.5, n)
    X = np.c_[t, np.sin(2 * np.pi * t / 52), np.cos(2 * np.pi * t / 52)]
    return X, y
