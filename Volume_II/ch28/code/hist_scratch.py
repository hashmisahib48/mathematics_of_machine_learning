import numpy as np


class HistStump:
    """Best split of one node (Section 28.3)."""

    def __init__(self, max_bins=255, l2=0.0, min_leaf=20):
        self.max_bins, self.l2 = max_bins, l2
        self.min_leaf = min_leaf

    def _bin(self, X):                         # Eqs. 28.3-1, 28.3-2
        self.thr_, cols = [], []
        for c in X.T:
            ok = ~np.isnan(c)
            v = np.unique(c[ok])
            if len(v) <= self.max_bins:    # few values: midpoints
                t = (v[:-1] + v[1:]) / 2
            else:
                q = np.linspace(0, 100, self.max_bins + 1)[1:-1]
                t = np.unique(np.percentile(
                    c[ok], q, method="averaged_inverted_cdf"))
            self.thr_.append(t)
            b = np.searchsorted(t, c)
            cols.append(np.where(ok, b, self.max_bins))  # NaN: last
        return np.column_stack(cols)

    def _gain(self, L, R):                     # Eq. 28.3-9
        # L, R = (sum g, sum h, count) of the two children
        if min(L[2], R[2]) < self.min_leaf:
            return -1.0
        if min(L[1], R[1]) < 1e-3:
            return -1.0
        s = lambda a: a[0] ** 2 / (a[1] + self.l2)
        return s(L) + s(R) - s(L + R)

    def fit(self, X, g, h):
        best, m = -1.0, self.max_bins + 1
        for j, c in enumerate(self._bin(X).T):
            hist = np.stack([np.bincount(c, g, m),     # Eq. 28.3-7
                             np.bincount(c, h, m),
                             np.bincount(c, minlength=m)])
            K = len(self.thr_[j]) + 1          # bins without NaN
            cum = np.cumsum(hist[:, :K], axis=1)       # Eq. 28.3-8
            miss = hist[:, -1]
            has = miss[2] > 0
            for nan_left in ((False, True) if has else (False,)):
                last = K if has and not nan_left else K - 1
                for k in range(last):          # bins 0..k go left
                    L = cum[:, k] + (miss if nan_left else 0)
                    R = cum[:, -1] + miss - L
                    gain = self._gain(L, R)
                    if gain > best:
                        best, self.j_, self.k_ = gain, j, k
                        self.nan_left_ = nan_left
                        self.sums_ = (L, R)
        self.gain_ = best
        if not np.isnan(X[:, self.j_]).any():  # no NaN in training:
            self.nan_left_ = self.sums_[0][2] > self.sums_[1][2]
        self.value_ = [-s[0] / (s[1] + self.l2)   # Eq. 28.3-6
                       for s in self.sums_]
        return self

    def predict(self, X):                  # leaf value of each row
        c = X[:, self.j_]
        t = self.thr_[self.j_]
        left = np.where(np.isnan(c), self.nan_left_,
                        np.searchsorted(t, c) <= self.k_)
        return np.where(left, *self.value_)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, make_friedman1
    from sklearn.ensemble import HistGradientBoostingClassifier as C
    from sklearn.ensemble import HistGradientBoostingRegressor as R

    def compare(name, mine, ref):
        n = ref._predictors[0][0].nodes[:3]
        thr = np.append(mine.thr_[mine.j_], np.inf)[mine.k_]
        a = [mine.gain_, mine.j_, thr, mine.nan_left_, *mine.value_]
        b = [n["gain"][0], n["feature_idx"][0],
             n["num_threshold"][0], n["missing_go_to_left"][0],
             n["value"][1], n["value"][2]]
        assert np.allclose(a, b, rtol=1e-4), name
        print(name)
        for who, v in (("mine   ", a), ("sklearn", b)):
            print(" ", who, " ".join("%9.4f" % u for u in v))

    print("            gain  feature thresh NaN-left  left  right")
    X, y = make_friedman1(2000, n_features=6, random_state=0)
    rng = np.random.default_rng(0)     # blank x4 where y is small
    low = (y < np.quantile(y, 0.3)) & (rng.random(2000) < 0.7)
    X[low, 3] = np.nan
    mine = HistStump(l2=3.0).fit(X, y.mean() - y, np.ones(2000))
    ref = R(max_iter=1, max_leaf_nodes=2, l2_regularization=3.0,
            learning_rate=1.0, early_stopping=False).fit(X, y)
    compare("regression, l2 = 3, NaN in x4", mine, ref)

    X, y = load_breast_cancer(return_X_y=True)
    p = y.mean()
    mine = HistStump().fit(X, p - y, np.full(len(y), p * (1 - p)))
    ref = C(max_iter=1, max_leaf_nodes=2, learning_rate=1.0,
            early_stopping=False).fit(X, y)
    compare("classification, breast cancer", mine, ref)
