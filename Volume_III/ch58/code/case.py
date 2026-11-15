# Breast cancer: what does the gradient-boosting model rely on?
import numpy as np
from bcdata import load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

Xa, Xb, ya, yb, names = load()
ym, yt = 1 - ya, 1 - yb                    # 1 = malignant
gb = GradientBoostingClassifier(random_state=0).fit(Xa, ym)
print("test accuracy", round(gb.score(Xb, yt), 4))
kw = dict(n_repeats=30, random_state=0)
acc = permutation_importance(gb, Xb, yt, **kw)
ll = permutation_importance(gb, Xb, yt, scoring="neg_log_loss",
                            **kw)
print(f"{'feature':22s} impurity acc drop ll rise  +- sd")
for j in np.argsort(-ll.importances_mean)[:6]:
    print(f"{names[j]:22s} {gb.feature_importances_[j]:7.3f}"
          f" {acc.importances_mean[j]:8.4f}"
          f" {ll.importances_mean[j]:7.4f}"
          f" {ll.importances_std[j]:7.4f}")

block = [0, 2, 3, 20, 22, 23]       # radius, perimeter, area
rng, drops = np.random.RandomState(0), []
for _ in range(30):                 # one shuffle for the block
    Xp = Xb.copy()
    Xp[:, block] = Xb[rng.permutation(len(yt))][:, block]
    drops.append(gb.score(Xb, yt) - gb.score(Xp, yt))
print("size block: drop", round(np.mean(drops), 4),
      " sum of singles",
      round(acc.importances_mean[block].sum(), 4))
