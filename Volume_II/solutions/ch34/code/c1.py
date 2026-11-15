# Problem 34C.1: harmonic values lie between the labels; they are not probabilities of the class.
import warnings
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelPropagation
warnings.simplefilter('ignore')
rng = np.random.default_rng(0)

print("== 5000 random graphs: harmonic values against the labels 0 and 1")
lo, hi, strict, n_bad, conn, conn_strict = 9., -9., 0, 0, 0, 0
for trial in range(5000):
    n = int(rng.integers(6, 16)); W = rng.random((n, n)) * (rng.random((n, n)) < 0.4); W = np.triu(W, 1); W = W + W.T
    L = np.arange(3); U = np.arange(3, n); fL = np.array([0., 1., 1.]) if trial % 2 else np.array([0., 0., 1.])
    d = W.sum(1); M = np.diag(d[U]) - W[np.ix_(U, U)]
    if np.linalg.cond(M) > 1e10: n_bad += 1; continue                  # an unlabelled component: skip
    f = np.linalg.solve(M, W[np.ix_(U, L)] @ fL)
    lo, hi = min(lo, f.min()), max(hi, f.max()); ins = bool((f > 1e-12).all() and (f < 1 - 1e-12).all()); strict += ins
    if connected_components(W > 0)[0] == 1: conn += 1; conn_strict += ins
print("graphs used %d, skipped (no path to a label) %d; smallest value %.4f, largest %.4f; all strictly inside (0, 1): %d"
      % (5000 - n_bad, n_bad, lo, hi, strict))
print("of them %d graphs are connected, and in %d of these every value is strictly inside (0, 1)" % (conn, conn_strict))

print("\n== two moons (Figure 34.2-1), one label per moon: the values change with gamma, the classes do not")
X, y = make_moons(200, noise=0.1, random_state=1)
i0 = np.where(y == 0)[0][np.argmin(X[y == 0, 0])]; i1 = np.where(y == 1)[0][np.argmax(X[y == 1, 0])]
yy = -np.ones(200, int); yy[[i0, i1]] = y[[i0, i1]]; unl = yy == -1
print(" gamma   min    max   mean(moon 0)  mean(moon 1)  accuracy  n_iter")
for g in (20, 50, 100):
    m = LabelPropagation(gamma=g, max_iter=400000, tol=1e-9).fit(X, yy); P = m.label_distributions_[:, 1]
    print(f"{g:6d} {P[unl].min():6.3f} {P[unl].max():6.3f} {P[unl][y[unl] == 0].mean():12.3f} {P[unl][y[unl] == 1].mean():13.3f} {np.mean(m.transduction_ == y):9.2f} {m.n_iter_:8d}")
