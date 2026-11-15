# The invented "returns" model of Chapter 46: 20 stocks in four sectors.
import numpy as np

def true_model(p=20, sectors=4, link=-0.45, cross=-0.25):
    """Sparse precision Theta0 (chain inside each sector, two cross links),
    stock volatilities in percent; returns (Sigma, Theta, sd)."""
    T0 = np.eye(p)
    size = p // sectors
    for s in range(sectors):
        for j in range(size - 1):
            a = s * size + j
            T0[a, a + 1] = T0[a + 1, a] = link
    for a, b in ((4, 5), (9, 10)):         # two links between sectors
        T0[a, b] = T0[b, a] = cross
    sd = np.linspace(0.8, 2.0, p)           # daily volatility, per cent
    S0 = np.linalg.inv(T0)
    Sigma = S0 * np.outer(sd, sd)
    Theta = T0 / np.outer(sd, sd)
    return Sigma, Theta, sd

def draw(n, seed, p=20):
    Sigma, _, _ = true_model(p)
    return np.random.RandomState(seed).multivariate_normal(np.zeros(p), Sigma, size=n)
