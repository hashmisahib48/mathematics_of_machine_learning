# Problem 28C.1: why the first tree has one leaf below 40 rows, and what to do with 150 rows
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import cross_val_score, KFold

def leaves(n, **kw):
    X, y = make_friedman1(n, n_features=5, random_state=0)
    m = HGBR(max_iter=1, early_stopping=False, **kw).fit(X, y)
    return m._predictors[0][0].get_n_leaf_nodes()

print("   n  leaves  floor(n/20)   leaves with min_samples_leaf=5  floor(n/5)")
for n in (30, 36, 39, 40, 41, 50, 59, 60, 80, 120, 150):
    print(f"{n:4d} {leaves(n):6d} {n // 20:11d} {leaves(n, min_samples_leaf=5):20d} {n // 5:14d}")

for n in (39, 40):
    X, y = make_friedman1(n, n_features=5, random_state=0)
    p = HGBR(early_stopping=False).fit(X, y)
    print(f"n = {n}: 100 trees, number of different predictions {len(np.unique(p.predict(X).round(10)))}, "
          f"mean of y {y.mean():.4f}, prediction range {p.predict(X).min():.4f} to {p.predict(X).max():.4f}")
print("\nFirst split of 40 rows: left sizes allowed with min_samples_leaf = 20:",
      [k for k in range(1, 40) if k >= 20 and 40 - k >= 20])

print("\n150 rows: five-fold R2 (mean of 5 data sets, 5 folds each)")
cfgs = {"defaults": {},
        "min_samples_leaf=5": dict(min_samples_leaf=5),
        "msl=5, 8 leaves": dict(min_samples_leaf=5, max_leaf_nodes=8),
        "msl=5, 8 leaves, l2=1": dict(min_samples_leaf=5, max_leaf_nodes=8, l2_regularization=1.0),
        "msl=5, 8 leaves, lr=0.05, 300 trees": dict(min_samples_leaf=5, max_leaf_nodes=8, learning_rate=0.05, max_iter=300),
        "msl=10, 4 leaves, lr=0.1": dict(min_samples_leaf=10, max_leaf_nodes=4)}
for name, kw in cfgs.items():
    s = []
    for seed in range(5):
        X, y = make_friedman1(150, n_features=5, noise=1.0, random_state=seed)
        s.append(cross_val_score(HGBR(early_stopping=False, random_state=0, **kw), X, y,
                                 cv=KFold(5, shuffle=True, random_state=0)).mean())
    print(f"{name:38s} R2 = {np.mean(s):.3f}  (data sets {min(s):.3f} to {max(s):.3f})")
