# Problem 54B.1: the leak against the number of features kept
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

n, p = 60, 1000
cv = StratifiedKFold(5, shuffle=True, random_state=0)
clf = LogisticRegression(max_iter=1000)
print("  k   first   inside   |r| of kept   |r| of all columns")
for k in (1, 10, 50, 200):
    first, inside, rr, ra = [], [], [], []
    for seed in range(20):          # 20 data sets of pure noise
        rng = np.random.RandomState(seed)
        X = rng.normal(size=(n, p))
        y = rng.randint(0, 2, n)    # labels unrelated to X
        sel = SelectKBest(f_classif, k=k).fit(X, y)
        Xk = sel.transform(X)       # chosen on all 60 rows
        first.append(cross_val_score(clf, Xk, y, cv=cv).mean())
        pipe = make_pipeline(SelectKBest(f_classif, k=k), clf)
        inside.append(cross_val_score(pipe, X, y, cv=cv).mean())
        F = sel.scores_[sel.get_support()]  # |r| from F
        rr.append(np.sqrt(F / (F + n - 2)).mean())
        Fa = sel.scores_                    # all 1000 columns
        ra.append(np.sqrt(Fa / (Fa + n - 2)).mean())
    print(f"{k:3d} {np.mean(first):7.3f} {np.mean(inside):8.3f}"
          f" {np.mean(rr):12.3f} {np.mean(ra):12.3f}")
print("chance correlation sqrt(2 ln(2p)/n) = %.3f" %
      np.sqrt(2 * np.log(2 * p) / n))
