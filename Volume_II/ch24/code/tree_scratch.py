# A classification tree from scratch: Gini, midpoint thresholds.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC


class MyTree:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)    # as scikit-learn
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.K_, self.nodes_ = len(self.classes_), []
        self._grow(X, yi, 0)
        return self

    def _gini(self, n_k):                       # Eq. 24.3-3
        p = n_k / n_k.sum(axis=-1, keepdims=True)
        return 1.0 - (p ** 2).sum(axis=-1)

    def _split(self, X, y):                     # Eq. 24.3-5
        n, m = len(y), self.min_samples_leaf
        onehot, best = np.eye(self.K_)[y], (np.inf, None, None)
        n_l = np.arange(1, n)
        for j in range(X.shape[1]):
            o = np.argsort(X[:, j], kind="stable")
            v = X[o, j].astype(float)
            c_l = np.cumsum(onehot[o], axis=0)[:-1]
            c_r = onehot.sum(axis=0) - c_l
            gl, gr = self._gini(c_l), self._gini(c_r)
            g = (n_l * gl + (n - n_l) * gr) / n
            ok = (v[1:] > v[:-1] + 1e-7)
            ok &= (n_l >= m) & (n - n_l >= m)
            k = np.argmin(np.where(ok, g, np.inf))
            if ok[k] and g[k] < best[0]:
                best = (g[k], j, v[k] / 2 + v[k + 1] / 2)
        return best[1:]

    def _grow(self, X, y, depth):
        n_k = np.bincount(y, minlength=self.K_)
        node = {"f": -2, "t": -2.0, "l": -1, "r": -1,
                "p": n_k / len(y)}
        self.nodes_.append(node)
        me = len(self.nodes_) - 1
        stop = (n_k.max() == len(y) or depth == self.max_depth
                or len(y) < 2 * self.min_samples_leaf)
        j, t = (None, None) if stop else self._split(X, y)
        if j is not None:
            left = X[:, j] <= t
            node.update(f=j, t=t)
            node["l"] = self._grow(X[left], y[left], depth + 1)
            node["r"] = self._grow(X[~left], y[~left], depth + 1)
        return me

    def predict_proba(self, X):
        out = []
        for x in np.asarray(X, dtype=np.float32):
            nd = self.nodes_[0]
            while nd["l"] != -1:
                nd = self.nodes_[nd["l"] if x[nd["f"]] <= nd["t"]
                                 else nd["r"]]
            out.append(nd["p"])
        return np.array(out)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    for depth in (1, 2, 3, None):
        mine = MyTree(max_depth=depth).fit(X_tr, y_tr)
        ref = DTC(max_depth=depth, random_state=0).fit(X_tr, y_tr)
        assert len(mine.nodes_) == ref.tree_.node_count
        same = np.array_equal(mine.predict(X_te), ref.predict(X_te))
        if depth is not None:
            assert same and np.allclose(
                mine.predict_proba(X_te), ref.predict_proba(X_te))
        acc = np.mean(mine.predict(X_te) == y_te)
        print(f"max_depth={depth}: {len(mine.nodes_)} nodes, test "
              f"accuracy {acc:.4f} vs {ref.score(X_te, y_te):.4f}; "
              f"same predictions: {same}")
