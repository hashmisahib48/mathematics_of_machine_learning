# TunedThresholdClassifierCV from scratch (cost objective, Eq. 58.3-4).
import numpy as np
from bcdata import load
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import (StratifiedKFold,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_FN, C_FP = 10, 1


def mean_cost(y, yes):                      # y = 1 means malignant
    return (C_FN * ((y == 1) & ~yes).sum()
            + C_FP * ((y == 0) & yes).sum()) / len(y)


def tune(model, X, y, n_thr=100, n_splits=5):
    folds = []
    for a, b in StratifiedKFold(n_splits).split(X, y):
        s = clone(model).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        t = np.linspace(s.min(), s.max(), n_thr)
        folds.append((t, [-mean_cost(y[b], s >= u) for u in t]))
    grid = np.linspace(min(f[0].min() for f in folds),
                       max(f[0].max() for f in folds), n_thr)
    mean = np.mean([np.interp(grid, t, c) for t, c in folds], axis=0)
    k = mean.argmax()                       # first maximum
    return grid[k], mean[k]


Xa, Xb, ya, yb, names = load()
ym = 1 - ya                                 # 1 = malignant
model = make_pipeline(StandardScaler(), LogisticRegression())
mine = tune(model, Xa, ym)
score = make_scorer(lambda y, p: mean_cost(y, p == 1),
                    greater_is_better=False)
lib = TunedThresholdClassifierCV(model, scoring=score).fit(Xa, ym)
print("scratch:", np.round(mine, 6))
print("library:", np.round([lib.best_threshold_, lib.best_score_], 6))
assert np.allclose(mine, [lib.best_threshold_, lib.best_score_])
