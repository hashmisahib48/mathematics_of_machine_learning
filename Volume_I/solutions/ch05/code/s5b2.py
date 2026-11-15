# Problem 5B.2: the scale of a column and the L-BFGS iterations.
import warnings
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=5,
                           random_state=0)


def fit(D):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LogisticRegression().fit(D, y)   # lbfgs, max_iter=100
    warned = any(w.category is ConvergenceWarning for w in caught)
    return m.n_iter_[0], warned, m.score(D, y), D.std(axis=0)


Xb = X.copy()
Xb[:, 0] *= 1000                      # first column, other units
Xs = StandardScaler().fit_transform(Xb)
for name, D in (("unmodified", X), ("column x 1000", Xb),
                ("x1000, standardised", Xs)):
    n_iter, warned, acc, sd = fit(D)
    print(f"{name:20s} n_iter_={n_iter:3d} warn={warned!s:5s}"
          f" sd ratio={sd.max() / sd.min():6.1f} acc={acc:.3f}")

print("\nfactor  n_iter_  warning")           # how far can it go?
for f in (1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7):
    Xf = X.copy()
    Xf[:, 0] *= f
    n_iter, warned, acc, sd = fit(Xf)
    print(f"{f:6.0e} {n_iter:8d}  {warned}")
