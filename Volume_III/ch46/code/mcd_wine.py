# A robust covariance on wine with twelve injected outliers.
import numpy as np
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.covariance import EmpiricalCovariance, MinCovDet

wine = load_wine()
cols = [0, 6, 9, 12]    # alcohol, flavanoids, colour, proline
X, y = wine.data[:, cols], wine.target
rng = np.random.RandomState(0)
bad = rng.choice(np.where(y == 2)[0], 12, replace=False)
W = np.vstack([X[y == 0], X[bad]])          # 59 clean + 12 injected
isbad = np.r_[np.zeros(59, bool), np.ones(12, bool)]
cut = chi2(4).isf(0.025)                    # the 97.5 % point
print("cut-off", cut.round(3), " n, p =", W.shape)

ref = EmpiricalCovariance().fit(W[~isbad])  # the target
mcd = MinCovDet(random_state=0).fit(W)
emp = EmpiricalCovariance().fit(W)
for name, m in [("empirical", emp), ("MinCovDet", mcd)]:
    flag = m.mahalanobis(W) > cut
    err = np.linalg.norm(m.covariance_ - ref.covariance_)
    err /= np.linalg.norm(ref.covariance_)
    print(f"{name:10s} caught {flag[isbad].sum():2d} of 12, "
          f"false alarms {flag[~isbad].sum():2d} of 59, "
          f"relative error {err:.3f}")
print("support:", mcd.raw_support_.sum(), "raw,",
      mcd.support_.sum(), "reweighted; injected in raw support:",
      (mcd.raw_support_ & isbad).sum())
print("location, clean    ", ref.location_.round(1))
print("location, empirical", emp.location_.round(1))
print("location, MinCovDet", mcd.location_.round(1))
print("dist_ equals mahalanobis:",
      np.allclose(mcd.dist_, mcd.mahalanobis(W)))
