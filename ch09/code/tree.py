# Classification and regression trees (CART) in plain Python.
from collections import Counter

def gini(y):
    n = len(y)
    return 1.0 - sum((k / n) ** 2 for k in Counter(y).values())

def mse(y):
    m = sum(y) / len(y)
    return sum((t - m) ** 2 for t in y) / len(y)

class _Tree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _best_split(self, X, y):                      # Algorithm 9.2
        n, best = len(y), (self.impurity(y), None, None)
        for j in range(len(X[0])):
            v = sorted(set(x[j] for x in X))
            for a, b in zip(v, v[1:]):                # Step 1
                s = (a + b) / 2
                L = [t for x, t in zip(X, y) if x[j] <= s]
                R = [t for x, t in zip(X, y) if x[j] > s]
                w = (len(L) * self.impurity(L)        # Step 2
                     + len(R) * self.impurity(R)) / n
                if w < best[0] - 1e-12:               # Step 3
                    best = (w, j, s)
        return best

    def _grow(self, X, y, depth):                     # Algorithm 9.1
        node = {"n": len(y), "value": self.leaf_value(y)}
        if (len(y) < self.min_samples_split or self.impurity(y) == 0
                or depth == self.max_depth):          # Step 1
            return node
        w, j, s = self._best_split(X, y)              # Step 2
        if j is None:
            return node
        left = [i for i, x in enumerate(X) if x[j] <= s]
        right = [i for i, x in enumerate(X) if x[j] > s]
        node.update(feature=j, threshold=s,           # Step 3
                    left=self._grow([X[i] for i in left],
                                    [y[i] for i in left], depth + 1),
                    right=self._grow([X[i] for i in right],
                                     [y[i] for i in right], depth + 1))
        return node

    def fit(self, X, y):
        self.tree_ = self._grow(X, y, 0)
        return self

    def predict(self, Z):
        out = []
        for z in Z:
            node = self.tree_
            while "feature" in node:                  # walk down
                go_left = z[node["feature"]] <= node["threshold"]
                node = node["left"] if go_left else node["right"]
            out.append(node["value"])
        return out

class DecisionTreeClassifier(_Tree):
    impurity = staticmethod(gini)

    @staticmethod
    def leaf_value(y):
        c = Counter(y)
        return max(sorted(c), key=c.get)              # majority class

class DecisionTreeRegressor(_Tree):
    impurity = staticmethod(mse)

    @staticmethod
    def leaf_value(y):
        return sum(y) / len(y)                        # mean of the leaf
