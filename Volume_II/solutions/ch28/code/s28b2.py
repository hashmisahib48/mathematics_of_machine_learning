# Problem 28B.2: the patience of early stopping
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.model_selection import train_test_split

X, y = make_classification(12000, 20, n_informative=6, flip_y=0.1,
                           random_state=0)
rows = {}
print(" k  trees  best tree  best val loss  trees after best")
for k in (3, 10, 30):
    m = HGBC(max_iter=500, n_iter_no_change=k,
             random_state=0).fit(X, y)
    v = -m.validation_score_       # validation log loss; v[0] = start
    best = int(np.argmin(v))
    print(f"{k:2d} {m.n_iter_:6d} {best:10d} {v[best]:14.4f}"
          f" {m.n_iter_ - best:12d}")
    rows[k] = (m, best)

# the 1200 validation rows that scikit-learn held out (stratified)
m0 = rows[3][0]
_, Xv, _, yv = train_test_split(X, y, test_size=0.1, stratify=y,
                                random_state=m0._random_seed)
def row_loss(m, t):          # loss of each validation row, t trees
    p = list(m.staged_predict_proba(Xv))[t - 1][:, 1]
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))
a, b = row_loss(*rows[3]), row_loss(*rows[30])
n = len(yv)
print(f"\n{n} validation rows; mean loss k=3: {a.mean():.4f}, "
      f"k=30: {b.mean():.4f}")
d = a - b
print(f"paired difference {d.mean():.4f}, standard error "
      f"{d.std(ddof=1) / np.sqrt(n):.4f}")
print(f"standard error of one mean loss: "
      f"{a.std(ddof=1) / np.sqrt(n):.4f}")
