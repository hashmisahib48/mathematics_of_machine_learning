# Invented data of Chapter 49: a skewed feature with one outlier,
# and a two-feature parcel table whose second column is in grams.
import numpy as np

def make_skewed(n=200, seed=3):
    """Log-normal values and one outlier (30.0). Returns shape (n, 1)."""
    x = np.random.RandomState(seed).lognormal(0.0, 0.7, n - 1)
    return np.append(x, 30.0).reshape(-1, 1)

def make_parcels(n=300, seed=1):
    """Class set by distance (km); weight (g) carries no class signal."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    km = rng.normal(3.0 + 4.0 * y, 1.3)
    g = rng.uniform(100, 900, n)
    return np.column_stack([km, g]), y
