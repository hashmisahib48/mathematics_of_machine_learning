# A simulated motor-insurance portfolio (no data are downloaded).
import numpy as np
import pandas as pd

TRUE = {"young": 0.80, "old": 0.30, "power_mid": 0.40,
        "power_high": 0.90, "density": 0.50}   # true log-effects


def make_portfolio(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    power = rng.choice([0, 1, 2], size=n, p=[0.5, 0.35, 0.15])
    X = pd.DataFrame({
        "young": np.maximum(0, 30 - age) / 10,   # decades below 30
        "old": np.maximum(0, age - 60) / 10,     # decades above 60
        "power_mid": (power == 1) * 1.0,
        "power_high": (power == 2) * 1.0,
        "density": rng.normal(size=n),           # standardised
    })
    expo = rng.uniform(0.1, 1.0, n)              # years insured
    beta = np.array(list(TRUE.values()))
    eta = np.log(0.06) + X.to_numpy() @ beta     # log-frequency
    count = rng.poisson(expo * np.exp(eta))      # number of claims
    size = rng.gamma(2.0, 750.0, size=count.sum())   # mean 1500
    owner = np.repeat(np.arange(n), count)
    amount = np.bincount(owner, weights=size, minlength=n)
    return X, expo, count, amount
