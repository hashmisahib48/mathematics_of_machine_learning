# The eight toy readings: LOF (k = 2) and isolation scores by hand.
import numpy as np
from sklearn.ensemble import IsolationForest

x = np.array([0, 1, 2, 3, 7, 13, 17, 21.])[:, None]   # mm/s
n, k = len(x), 2
D = np.abs(x - x.T)
np.fill_diagonal(D, np.inf)                 # a point is not its own
nn = np.argsort(D, axis=1)[:, :k]           # neighbour N_k(a)
dk = D[np.arange(n), nn[:, -1]]             # k-distance d_k(a)
rows = np.arange(n)[:, None]
reach = np.maximum(D[rows, nn], dk[nn])     # Eq. 48.4-4
lrd = 1 / reach.mean(1)                     # Eq. 48.4-5
lof = lrd[nn].mean(1) / lrd                 # Eq. 48.4-6
print("x    d_2   lrd      LOF")
for a in range(n):
    print(f"{x[a, 0]:3.0f} {dk[a]:5.1f} {lrd[a]:7.4f}"
          f" {lof[a]:7.4f}")

def c(m):                                   # Eq. 48.4-1 as coded
    return 0 if m <= 1 else 1 if m == 2 else \
        2 * (np.log(m - 1) + np.euler_gamma) - 2 * (m - 1) / m

f = IsolationForest(n_estimators=1000, max_samples=8,
                    random_state=0).fit(x)
x32, h = x.astype(np.float32), np.zeros(n)
for t in f.estimators_:                # h = edges + c(leaf size)
    edges = np.asarray(t.decision_path(x32).sum(1)).ravel() - 1
    size = t.tree_.n_node_samples[t.apply(x32)]
    h += edges + [c(m) for m in size]
Eh = h / 1000
s = 2 ** (-Eh / c(8))                       # Eq. 48.4-3
print("c(8) =", round(c(8), 4))
print("E h  :", Eh.round(2))
print("score:", s.round(3))
print("matches score_samples:", np.allclose(s, -f.score_samples(x)))
