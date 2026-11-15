# Problem 37B.2: the centroid linkage in the Lance-Williams loop.
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

# (a_i, a_j, beta, gamma) of the centroid row, works on d^2
def centroid(i, j, k):
    return i / (i + j), j / (i + j), -i * j / (i + j) ** 2, 0.0

def heights(X):
    D = squareform(pdist(X)) ** 2        # squared distances
    np.fill_diagonal(D, np.inf)
    size, alive, hs = np.ones(len(X)), list(range(len(X))), []
    while len(alive) > 1:
        sub = D[np.ix_(alive, alive)]
        a, b = divmod(sub.argmin(), len(alive))
        i, j = alive[a], alive[b]
        hs.append(D[i, j] ** 0.5)        # height = sqrt(d^2)
        rest = [k for k in alive if k not in (i, j)]
        ai, aj, be, _ = centroid(size[i], size[j], size[rest])
        new = ai * D[i, rest] + aj * D[j, rest] + be * D[i, j]
        D[i, rest] = D[rest, i] = new
        size[i] += size[j]; alive.remove(j)
    return np.array(hs)

finch = np.array([[3,3],[1,2],[0,2],[7,4],[11,2],[5,12],[8,9]], float)
for name, X in (("3 points", np.array([[0, 0], [2, 0], [1, 1.7]])),
                ("7 finches", finch)):
    mine = heights(X)
    ref = linkage(X, "centroid")[:, 2]
    print(name, "\n  mine :", np.round(mine, 4))
    print("  scipy:", np.round(ref, 4), "\n  equal:",
          np.allclose(mine, ref), " inversions:",
          int((np.diff(mine) < -1e-12).sum()))
