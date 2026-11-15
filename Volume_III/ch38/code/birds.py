# The data of the chapter: two crescent-shaped flocks and strays.
import numpy as np
from sklearn.datasets import make_moons


def birds():
    X, y = make_moons(n_samples=240, noise=0.06, random_state=0)
    rng = np.random.default_rng(1)
    S = rng.uniform([-1.4, -0.9], [2.4, 1.4], (12, 2))
    return np.r_[X, S], np.r_[y, np.full(12, -1)]


def varying(seed=3):
    """four groups of different densities and a little uniform noise"""
    rng = np.random.default_rng(seed)
    G = [((0.0, 0.0), 0.22, 70), ((1.5, 0.2), 0.22, 70),
         ((4.6, 2.0), 0.5, 45), ((5.6, 2.0), 0.5, 45),
         ((2.2, 3.2), 0.12, 40)]
    X = [rng.normal(c, s, (n, 2)) for c, s, n in G]
    y = [np.full(n, [0, 1, 2, 2, 3][k]) for k, (c, s, n) in enumerate(G)]
    X.append(rng.uniform([-1.5, -1.5], [7.0, 4.5], (20, 2)))
    y.append(np.full(20, -1))
    return np.vstack(X), np.concatenate(y)


def gps(seed=7):
    """GPS stops: two depots a street apart, an office, a market, noise"""
    rng = np.random.default_rng(seed)
    cen = np.array([[0, 0], [110, 0], [600, 300], [200, 800]])
    sd, ns = [15, 15, 45, 110], [120, 120, 200, 260]
    X = [rng.normal(c, s, (n, 2)) for c, s, n in zip(cen, sd, ns)]
    y = [np.full(n, k) for k, n in enumerate(ns)]
    X.append(np.c_[rng.uniform(-300, 900, 80), rng.uniform(-300, 1100, 80)])
    y.append(np.full(80, -1))
    return np.vstack(X), np.concatenate(y)
