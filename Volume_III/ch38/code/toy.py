# The ten birds of Section 38.1: DBSCAN by hand in NumPy.
import numpy as np
from sklearn.cluster import DBSCAN

X = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [1, 2],
              [6, 1], [7, 1], [7, 2], [8, 2], [4, 5]], float)
eps, m = 1.5, 4                       # radius and min_samples
D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))  # distances
nb = D <= eps                         # row i: N_eps(i), Eq. 38.3-1
count = nb.sum(1)                     # includes the point itself
core = count >= m                     # Eq. 38.3-2
print("count", count)
print("core ", core.astype(int))
c_m = np.sort(D, axis=1)[:, m - 1]    # core distance, Eq. 38.3-3
print("c_m  ", np.round(c_m, 3))
assert np.array_equal(core, c_m <= eps)
# border: not core, but within eps of a core point
border = ~core & (nb & core[None]).any(1)
noise = ~core & ~border
print("border", np.where(border)[0] + 1,
      "noise", np.where(noise)[0] + 1)
# clusters: connected components of the core points (Section 38.3)
lab = -np.ones(len(X), int)
k = 0
for i in np.where(core)[0]:
    if lab[i] >= 0:
        continue
    stack = [i]
    lab[i] = k
    while stack:
        j = stack.pop()
        for q in np.where(nb[j] & core)[0]:
            if lab[q] < 0:
                lab[q] = k
                stack.append(q)
    k += 1
for i in np.where(border)[0]:   # a border point joins the
    lab[i] = lab[nb[i] & core].min()   # cluster of nearby cores
print("labels", lab)

ref = DBSCAN(eps=eps, min_samples=m).fit(X)
assert np.array_equal(lab, ref.labels_)
assert np.array_equal(np.where(core)[0], ref.core_sample_indices_)
print("agrees with DBSCAN")
