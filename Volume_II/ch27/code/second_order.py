# The best root split of the next tree: squared error or gain?
import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier as GBC

data = load_breast_cancer()
X, y = data.data, data.target


def best_split(F, kind):
    p = expit(F)
    r, h = y - p, p * (1 - p)                  # r_i and h_i
    best = (-np.inf, None)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j])
        x, cr, ch = X[o, j], np.cumsum(r[o]), np.cumsum(h[o])
        cn = np.arange(1, len(y) + 1)
        for i in np.flatnonzero(x[:-1] < x[1:]):
            G, H, n = cr[-1], ch[-1], len(y)
            if kind == "squared error":          # Eq. 24.4-2
                s = cr[i]**2 / cn[i] + (G - cr[i])**2 / (n - cn[i])
            else:                                # Eq. 27.12-3
                s = cr[i]**2 / ch[i] + (G - cr[i])**2 / (H - ch[i])
            if s > best[0]:
                best = (s, (j, (x[i] + x[i + 1]) / 2))
    return best[1]


for stages in (1, 3):
    g = GBC(n_estimators=stages + 1, max_depth=3, random_state=0)
    g.fit(X, y)
    F = list(g.staged_decision_function(X))[stages - 1].ravel()
    t = g.estimators_[stages, 0].tree_
    root = data.feature_names[t.feature[0]]
    print(f"after {stages} stage(s), scikit-learn's next root:",
          root, round(t.threshold[0], 3))
    for kind in ("squared error", "second-order gain"):
        j, s = best_split(F, kind)
        print(f"   {kind:18s} {data.feature_names[j]:16s} {s:.3f}")
