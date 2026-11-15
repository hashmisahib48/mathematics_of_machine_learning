# Problem 46B.3: how many intruders can MinCovDet stand?
import numpy as np
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.covariance import EmpiricalCovariance, MinCovDet

wine = load_wine()
X = wine.data[:, [0, 6, 9, 12]]        # four features
y = wine.target
cut = chi2(4).isf(0.025)                     # 97.5 % point, 11.14
print("cut-off %.2f; 59 clean wines (cultivar 1)" % cut)
print("   k  share   empirical      MinCovDet     h  raw support")
for k in (4, 8, 12, 16, 20, 32, 48):
    rng = np.random.RandomState(0)
    bad = rng.choice(np.where(y == 2)[0], k, replace=False)
    W = np.vstack([X[y == 0], X[bad]])
    isbad = np.r_[np.zeros(59, bool), np.ones(k, bool)]
    fe = EmpiricalCovariance().fit(W).mahalanobis(W) > cut
    m = MinCovDet(random_state=0).fit(W)
    fm = m.dist_ > cut
    h = int(np.ceil((len(W) + 4 + 1) / 2))
    print(f"{k:4d} {k / len(W):6.2f} {fe[isbad].sum():4d} of {k:2d}"
          f" ({fe[~isbad].sum():2d}) {fm[isbad].sum():4d} of {k:2d}"
          f" ({fm[~isbad].sum():2d}) {h:4d} {m.raw_support_.sum():5d}"
          f" ({(m.raw_support_ & isbad).sum()})")
print("(in brackets: false alarms among the 59 clean wines)")
