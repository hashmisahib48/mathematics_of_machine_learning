# A binary soft-margin SVM solved by SMO (Sections 22.4 to 22.6).
import numpy as np


def smo(K, s, C, tol=1e-5):
    """Dual of Eq. 22.4-6; Gram matrix K, labels s = +-1."""
    Q = s[:, None] * s * K                       # Eq. 22.3-9
    a, G = np.zeros(len(s)), -np.ones(len(s))    # G = Q a - 1
    for it in range(100000):
        sG = -s * G
        up = ((s > 0) & (a < C)) | ((s < 0) & (a > 0))
        lo = ((s > 0) & (a > 0)) | ((s < 0) & (a < C))
        i = np.flatnonzero(up)[np.argmax(sG[up])]
        if sG[i] - sG[lo].min() < tol:           # KKT gap
            break
        q = np.maximum(K[i, i] + np.diag(K) - 2 * K[i], 1e-12)
        gain = np.where(lo & (sG < sG[i]), (sG[i] - sG)**2 / q, 0)
        j = np.argmax(gain)                      # second-order pick
        if s[i] != s[j]:                         # box, Eq. 22.6-5
            L, H = max(0, a[j] - a[i]), min(C, C + a[j] - a[i])
        else:
            L, H = max(0, a[i] + a[j] - C), min(C, a[i] + a[j])
        aj = np.clip(a[j] + s[j] * (sG[j] - sG[i]) / q[j], L, H)
        ai = a[i] + s[i] * s[j] * (a[j] - aj)    # Eq. 22.6-4
        G += Q[:, i] * (ai - a[i]) + Q[:, j] * (aj - a[j])
        a[i], a[j] = ai, aj
    return a, (sG[up].max() + sG[lo].min()) / 2, it   # a, b, steps


class MySVC:
    def __init__(self, C=1.0, kernel="linear", gamma=1.0, tol=1e-5):
        self.C, self.kernel = C, kernel
        self.gamma, self.tol = gamma, tol

    def _k(self, A, B):                          # Gram matrix
        if self.kernel == "linear":
            return A @ B.T
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1) - 2 * A @ B.T
        return np.exp(-self.gamma * d2)          # Eq. 22.5-3

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        s = np.where(y == self.classes_[1], 1.0, -1.0)
        K = self._k(X, X)
        a, self.b_, self.n_iter_ = smo(K, s, self.C, self.tol)
        self.support_ = np.flatnonzero(a > 1e-10)
        self.dual_coef_ = (a * s)[self.support_]
        self.sv_ = X[self.support_]
        return self

    def decision_function(self, X):              # Eq. 22.5-1
        K = self._k(np.asarray(X, float), self.sv_)
        return K @ self.dual_coef_ + self.b_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0) * 1]

from sklearn.svm import SVC                      # the test
from sklearn.datasets import make_moons

X, y = make_moons(150, noise=0.25, random_state=3)
for kern, C, g in (("linear", 1.0, 1.0), ("rbf", 3.0, 1.5)):
    mine = MySVC(C, kern, g).fit(X, y)
    ref = SVC(C=C, kernel=kern, gamma=g, tol=1e-5).fit(X, y)
    d = np.abs(mine.decision_function(X) - ref.decision_function(X))
    assert np.array_equal(mine.support_, np.sort(ref.support_))
    assert d.max() < 1e-3
    assert (mine.predict(X) == ref.predict(X)).all()
    print(kern, "| SVs:", len(mine.support_), "| max |f - f_svc| =",
          f"{d.max():.1e}")
    print("   b:", round(mine.b_, 5), "vs",
          round(ref.intercept_[0], 5), "| steps:", mine.n_iter_,
          "vs", ref.n_iter_[0])
