"""Diabetes with holes: MCAR, MAR and MNAR mechanisms made explicit."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

NAMES = load_diabetes().feature_names
HOLE = [2, 3, 6, 8]                      # bmi, bp, s3, s5 get holes
AGE = 0                                  # age drives the MAR holes

def split():
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=0)

def prob(driver, rate, strength):
    """P(missing) = expit(a + strength * z), mean over rows = rate."""
    z = (driver - driver.mean()) / driver.std()
    a = brentq(lambda a: expit(a + strength * z).mean() - rate, -30, 30)
    return expit(a + strength * z)

def punch(X, kind, rate, seed, strength=2.0):
    """Return a copy of X with NaN in the HOLE columns."""
    rng = np.random.RandomState(seed)
    Xn = X.copy()
    for j in HOLE:
        if kind == "MCAR":
            p = np.full(len(X), rate)
        elif kind == "MAR":                   # depends on observed age
            p = prob(X[:, AGE], rate, strength)
        else:                                 # MNAR: depends on own value
            p = prob(X[:, j], rate, strength)
        Xn[rng.uniform(size=len(X)) < p, j] = np.nan
    return Xn
