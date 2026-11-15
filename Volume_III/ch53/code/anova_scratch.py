# Section 53.8: ANOVA selection (Eq. 53.3-1, 53.3-4) from scratch.
import numpy as np
from scipy.stats import f as f_dist
from sklearn.datasets import load_wine
from sklearn.feature_selection import (SelectFdr, SelectKBest,
                                       f_classif)
from bcdata import load
class AnovaSelector:
    def __init__(self, k=None, alpha=None):
        self.k, self.alpha = k, alpha

    def fit(self, X, y):
        n, K, p = len(y), len(np.unique(y)), X.shape[1]
        ssb, ssw = np.zeros(p), np.zeros(p)
        for c in np.unique(y):
            Xc = X[y == c]
            ssb += len(Xc) * (Xc.mean(0) - X.mean(0)) ** 2
            ssw += ((Xc - Xc.mean(0)) ** 2).sum(0)
        self.scores_ = (ssb / (K - 1)) / (ssw / (n - K))
        self.pvalues_ = f_dist.sf(self.scores_, K - 1, n - K)
        self.support_ = np.zeros(p, dtype=bool)
        if self.k is not None:                       # k best
            self.support_[np.argsort(self.scores_)[-self.k:]] = True
        else:                                        # Eq. 53.3-4
            ps = np.sort(self.pvalues_)
            ok = ps[ps <= self.alpha * np.arange(1, p + 1) / p]
            if len(ok):
                self.support_ = self.pvalues_ <= ok.max()
        return self


Xa, Xb, ya, yb, names = load()
Xw, yw = load_wine(return_X_y=True)
for lab, X, y in (("cancer", Xa, ya), ("wine", Xw, yw)):
    mine = AnovaSelector(k=5).fit(X, y)
    ref = SelectKBest(f_classif, k=5).fit(X, y)
    assert np.allclose(mine.scores_, ref.scores_)
    assert np.allclose(mine.pvalues_, ref.pvalues_, rtol=1e-6)
    assert (mine.support_ == ref.get_support()).all()
    bh = AnovaSelector(alpha=0.001).fit(X, y)
    ref = SelectFdr(f_classif, alpha=0.001).fit(X, y)
    assert (bh.support_ == ref.get_support()).all()
    print(lab, "agrees; BH keeps", bh.support_.sum(),
          "of", len(bh.support_))
