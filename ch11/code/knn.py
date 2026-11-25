# k-nearest neighbors for regression and classification.
from collections import Counter


class _KNN:
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors, self.weights = n_neighbors, weights

    def fit(self, X, y):                       # store the training data
        self.X_, self.y_ = [list(r) for r in X], list(y)
        return self

    def _neighbors(self, z):
        d = [sum((a - b) ** 2 for a, b in zip(z, r)) ** 0.5   # Step 1
             for r in self.X_]
        idx = sorted(range(len(d)), key=lambda i: d[i])        # Step 2
        return [(d[i], self.y_[i]) for i in idx[:self.n_neighbors]]

    def _w(self, dist):                        # Step 3: vote weights
        if self.weights == "distance":
            return 1.0 / dist if dist > 0 else 1e12
        return 1.0


class KNeighborsRegressor(_KNN):
    def predict(self, Z):
        out = []
        for z in Z:
            nb = self._neighbors(z)
            w = [self._w(d) for d, _ in nb]
            s = sum(wi * t for wi, (_, t) in zip(w, nb))      # Step 4
            out.append(s / sum(w))
        return out


class KNeighborsClassifier(_KNN):
    def predict(self, Z):
        out = []
        for z in Z:
            votes = Counter()
            for d, c in self._neighbors(z):
                votes[c] += self._w(d)                         # Step 4
            out.append(max(sorted(votes), key=votes.get))
        return out

    def score(self, Z, y):
        p = self.predict(Z)
        return sum(a == b for a, b in zip(p, y)) / len(y)
