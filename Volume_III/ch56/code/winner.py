# Section 56.8: optimism of best_score_ on twelve re-splits of the
# breast-cancer data (writes verify/winner.json).
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from common import svc_pipe

X, y = load_breast_cancer(return_X_y=True)
grid = {"svc__C": 2.0 ** np.arange(-3, 10, 3),
        "svc__gamma": 2.0 ** np.arange(-13, 0, 3)}     # 25 cells
rows = []
for seed in range(12):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    gs = GridSearchCV(svc_pipe(), grid, cv=5).fit(Xtr, ytr)
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    nested = cross_val_score(GridSearchCV(svc_pipe(), grid, cv=5),
                             Xtr, ytr, cv=outer).mean()
    rows.append([gs.best_score_, nested, gs.score(Xte, yte)])
    print(f"seed {seed:2d}  best_score_ {rows[-1][0]:.4f}"
          f"  nested {nested:.4f}  fresh {rows[-1][2]:.4f}")
R = np.array(rows)
print("means      ", np.round(R.mean(axis=0), 4))
print("best_score_ minus fresh:", round((R[:, 0] - R[:, 2]).mean(), 4),
      " nested minus fresh:", round((R[:, 1] - R[:, 2]).mean(), 4))
json.dump(R.tolist(), open("verify/winner.json", "w"))
