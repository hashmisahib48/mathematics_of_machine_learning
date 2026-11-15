# Problem 27B.1: absolute-error boosting from scratch.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as Tree
from sklearn.utils import check_random_state

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)
lr, f0 = 0.1, np.median(ytr)              # start: median of y
low = lambda e: np.percentile(e, 50, method="inverted_cdf")

rng, F, stages = check_random_state(0), np.full(len(ytr), f0), []
for m in range(100):
    r = ytr - F                           # residuals
    tree = Tree(max_depth=2, random_state=rng).fit(Xtr, np.sign(r))
    leaf = tree.apply(Xtr)                # tree fitted to the signs
    g = np.zeros(tree.tree_.node_count)
    for t in np.unique(leaf):             # leaf value: lower median
        g[t] = low(r[leaf == t])
    F += lr * g[leaf]
    stages.append((tree, g))

pred = np.full(len(Xte), f0)
for tree, g in stages:
    pred += lr * g[tree.apply(Xte)]
ref = GBR(loss="absolute_error", n_estimators=100, max_depth=2,
          random_state=0).fit(Xtr, ytr)
gap = np.abs(pred - ref.predict(Xte)).max()
print("largest difference:", gap)
print("test R^2 :", round(ref.score(Xte, yte), 4))
print("test MAE :", round(np.abs(yte - pred).mean(), 2))
