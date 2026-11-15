# K-means from scratch: Lloyd's algorithm with k-means++ seeding.
import numpy as np

def sq_dist(X, M):              # squared distances, shape (n, K)
    return ((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)

def seed_pp(X, K, rng):         # k-means++, one trial (Eq. 36.5-1)
    idx = [rng.integers(len(X))]
    d2 = sq_dist(X, X[idx])[:, 0]
    for _ in range(K - 1):
        i = rng.choice(len(X), p=d2 / d2.sum())
        idx.append(i)
        d2 = np.minimum(d2, sq_dist(X, X[[i]])[:, 0])
    return X[idx].copy()

class MyKMeans:
    def __init__(self, n_clusters=3, init="k-means++", max_iter=300,
                 tol=1e-4, random_state=0):
        self.n_clusters, self.init = n_clusters, init
        self.max_iter, self.tol = max_iter, tol
        self.random_state = random_state

    def fit(self, X):
        X, K = np.asarray(X, float), self.n_clusters
        rng = np.random.default_rng(self.random_state)
        mu = (seed_pp(X, K, rng) if isinstance(self.init, str)
              else np.array(self.init, float))
        tol = self.tol * X.var(axis=0).mean()  # scikit-learn's rule
        old = None
        for it in range(self.max_iter):
            lab = sq_dist(X, mu).argmin(axis=1)      # Eq. 36.4-1
            new = np.array([X[lab == k].mean(axis=0)
                            for k in range(K)])      # Eq. 36.4-2
            shift, mu = ((new - mu) ** 2).sum(), new
            strict = old is not None and np.array_equal(lab, old)
            if strict or shift <= tol:
                break
            old = lab
        if not strict:                  # labels must match centres
            lab = sq_dist(X, mu).argmin(axis=1)
        self.cluster_centers_, self.labels_ = mu, lab
        self.n_iter_ = it + 1
        d2 = sq_dist(X, mu)[np.arange(len(X)), lab]
        self.inertia_ = float(d2.sum())              # Eq. 36.3-1
        return self
