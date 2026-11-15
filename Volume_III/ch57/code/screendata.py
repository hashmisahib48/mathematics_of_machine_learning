# The invented screening test used throughout Chapter 57.
import numpy as np
from scipy.stats import norm


def make_screen(n=2000, prevalence=0.05, d=2.0, seed=0):
    """Scores of a screening test: healthy N(0, 1), ill N(d, 1)."""
    rng = np.random.RandomState(seed)
    n_ill = int(round(n * prevalence))
    y = np.r_[np.ones(n_ill, int), np.zeros(n - n_ill, int)]
    s = np.r_[rng.normal(d, 1.0, n_ill),
              rng.normal(0.0, 1.0, n - n_ill)]
    return s, y


def auc_binormal(d=2.0):
    """Exact AUC of N(d, 1) against N(0, 1): Phi(d / sqrt 2)."""
    return norm.cdf(d / np.sqrt(2))
