# Few samples, many features: what shrinkage buys (Listing 18.5).
import warnings
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA)

P, N, REPEATS = 50, 40, 20             # features, training size
rng = np.random.default_rng(0)
idx = np.arange(P)
L = np.linalg.cholesky(0.6 ** np.abs(idx[:, None] - idx[None, :]))
delta = 0.3 * (-1.0) ** idx             # means differ by delta


def draw(n):
    y = np.arange(n) % 2
    return rng.normal(size=(n, P)) @ L.T + y[:, None] * delta, y


lsqr = dict(solver="lsqr")
models = {"svd (default)": LDA(),
          "lsqr, no shrinkage": LDA(**lsqr),
          "lsqr, shrinkage 0.5": LDA(**lsqr, shrinkage=0.5),
          "lsqr, shrinkage auto": LDA(**lsqr, shrinkage="auto")}
acc = {name: [] for name in models}
warnings.simplefilter("ignore")
for _ in range(REPEATS):
    X_tr, y_tr = draw(N)
    X_te, y_te = draw(4000)
    for name, m in models.items():
        acc[name].append(m.fit(X_tr, y_tr).score(X_te, y_te))
print(f"p = {P} features, n = {N} training samples")
for name, a in acc.items():
    print(f"{name:21s} test accuracy {np.mean(a):.4f}")
try:
    LDA(solver="eigen").fit(X_tr, y_tr)
except np.linalg.LinAlgError as err:
    print("eigen, no shrinkage:", type(err).__name__)
