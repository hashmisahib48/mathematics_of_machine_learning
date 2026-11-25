import random

def sqdist(a, b):
    return sum((u - v) ** 2 for u, v in zip(a, b))

def kmeans_pp(X, k, rng):                             # Algorithm 10.2
    C = [rng.choice(X)]                                # Step 1
    while len(C) < k:
        D = [min(sqdist(x, c) for c in C) for x in X]  # Step 2
        C.append(rng.choices(X, weights=D)[0])         # Step 3
    return [list(c) for c in C]

class KMeans:
    def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-6, random_state=None):
        self.k, self.init, self.n_init = n_clusters, init, n_init
        self.max_iter, self.tol = max_iter, tol
        self.rng = random.Random(random_state)

    def _lloyd(self, X, C):                           # Algorithm 10.1
        for it in range(1, self.max_iter + 1):
            lab = [min(range(self.k), key=lambda j: sqdist(x, C[j]))
                   for x in X]                         # Step 2
            new = []
            for j in range(self.k):                    # Step 3
                M = [x for x, l in zip(X, lab) if l == j] or [C[j]]
                new.append([sum(col) / len(M) for col in zip(*M)])
            shift = max(sqdist(a, b) for a, b in zip(C, new))
            C = new
            if shift <= self.tol:                      # Step 4
                break
        lab = [min(range(self.k), key=lambda j: sqdist(x, C[j]))
               for x in X]
        J = sum(sqdist(x, C[l]) for x, l in zip(X, lab))
        return C, lab, J, it

    def fit(self, X):
        best = None
        starts = ([self.init] if not isinstance(self.init, str) else
                  [kmeans_pp(X, self.k, self.rng)
                   for _ in range(self.n_init)])       # Step 1
        for C0 in starts:
            r = self._lloyd(X, [list(c) for c in C0])
            if best is None or r[2] < best[2]:         # keep the best
                best = r
        (self.cluster_centers_, self.labels_, self.inertia_,
         self.n_iter_) = best
        return self

    def predict(self, Z):
        return [min(range(self.k), key=lambda j:
                    sqdist(z, self.cluster_centers_[j])) for z in Z]
