# Permutation importance and partial dependence from scratch.
import numpy as np
from bcdata import load
from scipy.stats.mstats import mquantiles
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.inspection import (partial_dependence,
                                permutation_importance)

def perm_importance(model, X, y, n_repeats=5, seed=0):
    s = np.random.RandomState(seed).randint(2**31)  # one seed
    base = model.score(X, y)
    out = np.empty((X.shape[1], n_repeats))
    for j in range(X.shape[1]):         # each column replays it
        rng, order = np.random.RandomState(s), np.arange(len(y))
        Xp = X.copy()
        for r in range(n_repeats):
            rng.shuffle(order)
            Xp[:, j] = Xp[order, j]     # shuffle column j
            out[j, r] = base - model.score(Xp, y)  # Eq. 58.4-1
    return out

Xa, Xb, ya, yb, names = load()
gb = GBC(random_state=0).fit(Xa, 1 - ya)        # 1 = malignant
mine = perm_importance(gb, Xb, 1 - yb, n_repeats=10)
lib = permutation_importance(gb, Xb, 1 - yb, n_repeats=10,
                             random_state=0)
d1 = np.abs(mine - lib.importances).max()

def pd_one(model, X, j, n_grid=100, pct=(0.05, 0.95)):
    lo, hi = mquantiles(X[:, j], prob=pct)
    grid = np.linspace(lo, hi, n_grid)  # grid of Eq. 58.4-3
    Xv, avg = X.copy(), []
    for v in grid:
        Xv[:, j] = v                    # set x_j = v
        avg.append(model.predict_proba(Xv)[:, 1].mean())
    return grid, np.array(avg)

g, avg = pd_one(gb, Xb, 22)                     # worst perimeter
ref = partial_dependence(gb, Xb, [22], method="brute",
                         response_method="predict_proba")
d2 = np.abs(avg - ref["average"][0]).max()
print("max |difference|: importances", d1, " PD", d2)
assert d1 < 1e-12 and d2 < 1e-12
