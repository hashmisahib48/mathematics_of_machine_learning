# Test of GraphSSL against scikit-learn (Section 34.7).
import numpy as np
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from scratch import GraphSSL

X, y = make_moons(150, noise=0.1, random_state=1)
rng = np.random.default_rng(0)
yy = -np.ones(150, int)
for k in (0, 1):                         # three labels per moon
    yy[rng.choice(np.where(y == k)[0], 3, replace=False)] = k
Xn = rng.uniform(-1.5, 2.5, (40, 2))     # 40 new points
kw = dict(max_iter=200000, tol=1e-13)
pairs = [("propagation", GraphSSL("propagation", gamma=10),
          LabelPropagation(gamma=10, **kw)),
         ("spreading", GraphSSL("spreading", gamma=10, alpha=0.7),
          LabelSpreading(gamma=10, alpha=0.7, **kw))]
for name, mine, ref in pairs:
    mine.fit(X, yy)
    ref.fit(X, yy)
    dF = abs(mine.label_distributions_ - ref.label_distributions_)
    dP = abs(mine.predict_proba(Xn) - ref.predict_proba(Xn))
    assert (mine.transduction_ == ref.transduction_).all()
    assert (mine.predict(Xn) == ref.predict(Xn)).all()
    assert dF.max() < 1e-8 and dP.max() < 1e-8
    acc = np.mean(mine.transduction_ == y)
    print(f"{name:11s} max|dF| {dF.max():.0e}  max|dP| "
          f"{dP.max():.0e}  accuracy {acc:.3f}  "
          f"n_iter_ {ref.n_iter_}")
