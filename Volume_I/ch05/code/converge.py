# Did the solver converge?  Warning, n_iter_, max_iter, tol.
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xs = StandardScaler().fit_transform(X)


def fit(data, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LogisticRegression(**kw).fit(data, y)
    warned = any(w.category is ConvergenceWarning for w in caught)
    return m, warned


print("LogisticRegression, lbfgs, tol = 1e-4")
for name, data in (("raw", X), ("scaled", Xs)):
    for max_iter in (100, 5000):
        m, warned = fit(data, max_iter=max_iter)
        print(f"  {name:6s} max_iter {max_iter:4d}: n_iter_ "
              f"{m.n_iter_[0]:4d}  ConvergenceWarning {warned}")

ref, _ = fit(Xs, tol=1e-12, max_iter=5000)
print("scaled data, effect of tol:")
for tol in (1e-1, 1e-2, 1e-4, 1e-6):
    m, _ = fit(Xs, tol=tol, max_iter=5000)
    err = np.abs(m.coef_ - ref.coef_).max()
    print(f"  tol {tol:.0e}: n_iter_ {m.n_iter_[0]:3d}"
          f"  max|w - w_ref| {err:.1e}")

Xd, yd = load_diabetes(return_X_y=True)
print("Lasso(alpha=0.01) on diabetes, max_iter = 1000")
for tol in (1e-2, 1e-4, 1e-6, 1e-8):
    m = Lasso(alpha=0.01, tol=tol).fit(Xd, yd)
    print(f"  tol {tol:.0e}: n_iter_ {m.n_iter_}")
