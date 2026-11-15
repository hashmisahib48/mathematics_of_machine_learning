# DBSCAN from scratch: the class (Eqs. 38.3-1 to 38.3-3) and a test.
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN
from birds import birds


class MyDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X):
        X = np.asarray(X, float)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        nb = [np.flatnonzero(r <= self.eps) for r in D]   # 38.3-1
        core = np.array([len(a) >= self.min_samples for a in nb])
        labels = np.full(len(X), -1)
        k = 0
        for i in np.flatnonzero(core):        # in index order
            if labels[i] != -1:
                continue
            labels[i] = k                     # a new cluster
            queue = deque([i])
            while queue:                      # breadth-first search
                j = queue.popleft()
                if not core[j]:
                    continue                  # a border point stops
                for q in nb[j]:
                    if labels[q] == -1:
                        labels[q] = k
                        queue.append(q)
            k += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(core)
        self.components_ = X[core]
        return self


X, _ = birds()                 # 252 birds (Section 38.1)
for eps, m in [(0.1, 3), (0.2, 5), (0.2, 1)]:
    a = MyDBSCAN(eps, m).fit(X)
    b = DBSCAN(eps=eps, min_samples=m).fit(X)
    assert np.array_equal(a.labels_, b.labels_)
    assert np.array_equal(a.core_sample_indices_, b.core_sample_indices_)
    print(f"eps={eps} m={m}: clusters={a.labels_.max() + 1},"
          f" noise={(a.labels_ == -1).sum()}, identical")
