# Shrinkage estimators from scratch: Ledoit-Wolf and OAS.
import numpy as np
from returns import draw
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance

def shrunk(S, a):                           # Eq. 46.4-1
    mu = np.trace(S) / len(S)
    return (1 - a) * S + a * mu * np.eye(len(S))

def lw_alpha(X):                            # Eq. 46.4-4
    n, p = X.shape
    D = X - X.mean(0)
    S = D.T @ D / n
    mu = np.trace(S) / p
    delta = ((S - mu * np.eye(p)) ** 2).sum() / p
    q4 = ((D**2).sum(1) ** 2).sum()         # sum of ||x_i||^4
    beta = (q4 / n - (S**2).sum()) / (n * p)
    return min(beta, delta) / delta, S

def oas_alpha(X):                           # Eq. 46.4-5
    n, p = X.shape
    S = np.cov(X.T, bias=True)
    mu = np.trace(S) / p
    al = (S**2).mean()
    return min((al + mu**2) / ((n + 1) * (al - mu**2 / p)), 1.0), S

for n in (30, 100):
    X = draw(n, seed=4)
    a, S = lw_alpha(X)
    ref = LedoitWolf().fit(X)
    assert np.isclose(a, ref.shrinkage_)
    assert np.allclose(shrunk(S, a), ref.covariance_)
    b, _ = oas_alpha(X)
    assert np.isclose(b, OAS().fit(X).shrinkage_)
    fixed = ShrunkCovariance(shrinkage=0.25).fit(X).covariance_
    assert np.allclose(shrunk(S, 0.25), fixed)
    gap = np.abs(shrunk(S, a) - ref.covariance_).max()
    print(f"n={n:4d}  LW alpha {a:.4f}  OAS alpha {b:.4f}  "
          f"max difference {gap:.1e}")
