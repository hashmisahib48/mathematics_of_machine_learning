# Naive agglomerative clustering with Lance-Williams updates.
import numpy as np
from scipy.spatial.distance import pdist, squareform

# (a_i, a_j, beta, gamma) of Eq. 37.3-6; sizes n_i, n_j, n_k
LW = {
    "single": lambda i, j, k: (.5, .5, 0, -.5),
    "complete": lambda i, j, k: (.5, .5, 0, .5),
    "average": lambda i, j, k: (i / (i + j), j / (i + j), 0, 0),
    "ward": lambda i, j, k: ((i + k) / (i + j + k),
                             (j + k) / (i + j + k),
                             -k / (i + j + k), 0),
}

class Agglo:
    def __init__(self, linkage="ward", n_clusters=2):
        self.linkage, self.n_clusters = linkage, n_clusters

    def fit(self, X):
        X = np.asarray(X, float); n = len(X)
        w = self.linkage == "ward"
        D = squareform(pdist(X)) ** (2 if w else 1)
        np.fill_diagonal(D, np.inf)
        size, ident, row = np.ones(n), np.arange(n), np.arange(n)
        alive, ch, hs = list(range(n)), [], []
        for step in range(n - 1):
            sub = D[np.ix_(alive, alive)]
            a, b = divmod(sub.argmin(), len(alive))
            i, j = alive[a], alive[b]        # first pair on ties
            d = D[i, j]
            ch.append(sorted((ident[i], ident[j])))
            hs.append(d ** 0.5 if w else d)
            rest = [k for k in alive if k not in (i, j)]
            ai, aj, be, ga = LW[self.linkage](size[i], size[j],
                                              size[rest])
            di, dj = D[i, rest], D[j, rest]
            new = ai * di + aj * dj + be * d + ga * abs(di - dj)
            D[i, rest] = D[rest, i] = new            # Eq. 37.3-6
            size[i] += size[j]; ident[i] = n + step
            row[row == j] = i; alive.remove(j)
            if len(alive) == self.n_clusters:        # cut the tree
                u = np.unique(row, return_inverse=True)
                self.labels_ = u[1]
        self.children_, self.distances_ = np.array(ch), np.array(hs)
        return self
