# Hand checks for Problems 59A.1 and 59A.2 (every number quoted in the solutions)
import numpy as np
from scipy.stats import binomtest, loguniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.model_selection import (RandomizedSearchCV, RepeatedKFold, KFold,
                                     cross_val_score)

print("== 59A.1  the bill of a search")
n_cand, k_in, rep, k_out = 40, 5, 2, 5
splits = k_in * rep
print("splits per candidate:", splits, " fits of the search:", n_cand * splits,
      " plus refit:", n_cand * splits + 1)
print("nested:", k_out, "x", n_cand * splits + 1, "=", k_out * (n_cand * splits + 1))
print("wrong answers: without refit", n_cand * splits, "/", k_out * n_cand * splits,
      "; forgetting the repeats", n_cand * k_in + 1, "/", k_out * (n_cand * k_in + 1))

COUNT = [0]
class CountingRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        COUNT[0] += 1
        self.w_ = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ y)
        return self
    def predict(self, X): return X @ self.w_

X, y = make_regression(200, 5, noise=1.0, random_state=0)
search = RandomizedSearchCV(CountingRidge(), {"alpha": loguniform(1e-3, 1e3)}, n_iter=n_cand,
                            cv=RepeatedKFold(n_splits=k_in, n_repeats=rep, random_state=0),
                            random_state=0)
search.fit(X, y); one = COUNT[0]; COUNT[0] = 0
cross_val_score(search, X, y, cv=KFold(k_out)); nested = COUNT[0]
print("counted: search", one, " nested", nested)
print("at 2 s per fit: search %.1f min, nested %.1f h" % (one * 2 / 60, nested * 2 / 3600))

print("\n== 59A.2  an interval for 12 errors in 143")
n, e = 143, 12
a = 1 - e / n
se = np.sqrt(a * (1 - a) / n)
h = 1.96 * se
print("correct", n - e, " accuracy %.5f" % a)
print("a(1-a) = %.5f   /n = %.6f   se = %.5f" % (a * (1 - a), a * (1 - a) / n, se))
print("1.96 se = %.5f   interval %.4f to %.4f" % (h, a - h, a + h))
print("half-width in patients: %.2f   one patient = %.5f" % (h * n, 1 / n))
w = binomtest(n - e, n).proportion_ci(confidence_level=0.95, method="wilson")
print("Wilson %.4f to %.4f" % (w.low, w.high))
rng = np.random.RandomState(0); c = np.r_[np.ones(n - e), np.zeros(e)]
bs = np.array([c[rng.randint(0, n, n)].mean() for _ in range(5000)])
print("bootstrap %.4f to %.4f" % tuple(np.percentile(bs, [2.5, 97.5])))
# compare with the calibrated SVC of Case A: 137/143
a2 = 137 / n; se2 = np.sqrt(a2 * (1 - a2) / n)
print("SVC of Case A: %.4f, interval %.4f to %.4f; difference of the two accuracies %.4f"
      % (a2, a2 - 1.96 * se2, a2 + 1.96 * se2, a2 - a))
sd = np.sqrt(se ** 2 + se2 ** 2)
print("se of the difference (independent approx) %.4f, ratio %.2f" % (sd, (a2 - a) / sd))
