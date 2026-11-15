# Checks behind the parameter map: n_init, tol, max_iter, weights.
import contextlib, io, warnings
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_wine().data)

def runs(est):                  # count the initialisations that ran
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        est.set_params(verbose=1).fit(X)
    lines = buf.getvalue().splitlines()
    return sum("Initialization complete" in s
               or (s.startswith("Init ") and "with method" in s)
               for s in lines)

arr = X[:3].copy()
print("number of runs when n_init='auto':")
def first_k(X, k, random_state):    # a callable initialiser
    return X[:k].copy()

for name, ini in (("k-means++", "k-means++"), ("random", "random"),
                  ("array", arr), ("callable", first_k)):
    k1 = runs(KMeans(3, init=ini, random_state=0))
    k2 = runs(MiniBatchKMeans(3, init=ini, random_state=0))
    print(f"  {name:10s} KMeans {k1:2d}   MiniBatchKMeans {k2:2d}")

print("tol (relative to the mean feature variance), same start:")
print("  mean variance:", X.var(axis=0).mean())
for tol in (1e-1, 1e-2, 1e-4, 1e-8, 0.0):
    km = KMeans(3, init=arr, n_init=1, tol=tol).fit(X)
    print(f"  tol={tol:<7g} iterations {km.n_iter_:2d}"
          f"  inertia {km.inertia_:.4f}")
print("max_iter, same start (no warning is raised):")
for mi in (1, 2, 3, 300):
    km = KMeans(3, init=arr, n_init=1, max_iter=mi).fit(X)
    print(f"  max_iter={mi:<4d} inertia {km.inertia_:.4f}")

w = np.arange(1, 7)             # integer weights = repeated rows
Xs = X[:6]
a = KMeans(2, init=Xs[:2], n_init=1).fit(Xs, sample_weight=w)
b = KMeans(2, init=Xs[:2], n_init=1).fit(np.repeat(Xs, w, axis=0))
print("weights equal repetition:",
      np.allclose(a.cluster_centers_, b.cluster_centers_),
      np.isclose(a.inertia_, b.inertia_))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    dup = np.array([[0.0], [0.0], [1.0], [1.0]])
    KMeans(3, n_init=1, random_state=0).fit(dup)
print("duplicates, K=3 on 2 distinct points:",
      [c.category.__name__ for c in caught])
