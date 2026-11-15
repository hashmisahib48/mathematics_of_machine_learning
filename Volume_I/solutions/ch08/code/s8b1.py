# Problem 8B.1: k-fold cross-validation written by hand
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


def cv_scores(model, X, y, k=5):
    """R2 on each held-out fold of KFold(k), without shuffling."""
    scores = []
    for tr, te in KFold(k).split(X):
        m = clone(model).fit(X[tr], y[tr])      # fresh copy per fold
        scores.append(r2_score(y[te], m.predict(X[te])))
    return np.array(scores)


X, y = load_diabetes(return_X_y=True)
mine = cv_scores(LinearRegression(), X, y)
ref = cross_val_score(LinearRegression(), X, y, cv=5)
print("cv_scores       :", mine.round(3), "mean %.3f" % mine.mean())
print("cross_val_score :", ref.round(3), "mean %.3f" % ref.mean())
print("largest difference:", np.abs(mine - ref).max())
print("same to 1e-12   :", np.allclose(mine, ref, atol=1e-12))
print("fold sizes      :", [len(te) for _, te in KFold(5).split(X)])
print("sd of the folds : %.3f" % mine.std(ddof=1))

fit = LinearRegression().fit(X, y)              # not cross-validated
print("R2 on its own training rows: %.3f" % fit.score(X, y))
sh = cv_scores(LinearRegression(), X[::-1], y[::-1])   # rows reversed
print("rows reversed   :", sh.round(3), "mean %.3f" % sh.mean())
