# Local outlier factor from scratch (Eq. 48.4-4 to 48.4-6).
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sensors import make_sensors

class LOF:
    def __init__(self, k=20):
        self.k = k
    def _knn(self, Q, own):             # k nearest training points
        D = np.sqrt(((Q[:, None] - self.X_[None]) ** 2).sum(2))
        if own:
            np.fill_diagonal(D, np.inf)     # not its own neighbour
        idx = np.argsort(D, axis=1)[:, :self.k]
        return np.take_along_axis(D, idx, 1), idx

    def _lrd(self, d, idx):                 # Eq. 48.4-4 and 48.4-5
        reach = np.maximum(d, self.kdist_[idx])
        return 1 / (reach.mean(1) + 1e-10)  # the library adds 1e-10

    def fit(self, X):
        self.X_ = X
        d, idx = self._knn(X, own=True)
        self.kdist_ = d[:, -1]              # d_k(a)
        self.lrd_ = self._lrd(d, idx)
        ratio = self.lrd_[idx] / self.lrd_[:, None]
        self.negative_outlier_factor_ = -ratio.mean(1)  # 48.4-6
        return self

    def score_samples(self, Q):         # new points, novelty mode
        d, idx = self._knn(Q, own=False)
        lrd = self._lrd(d, idx)
        return -(self.lrd_[idx] / lrd[:, None]).mean(1)
X = make_sensors()[0]
sc = StandardScaler().fit(X)
Z = sc.transform(X)
new = sc.transform([[56, 1.1], [66, 1.8], [70, 4.5]])
mine = LOF(20).fit(Z)
ref = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z)
assert np.allclose(mine.negative_outlier_factor_,
                   ref.negative_outlier_factor_)
assert np.allclose(mine.score_samples(new), ref.score_samples(new))
print("training LOF agrees; largest difference", np.abs(
    mine.negative_outlier_factor_ - ref.negative_outlier_factor_).max())
print("new readings (-LOF):", mine.score_samples(new).round(3))
