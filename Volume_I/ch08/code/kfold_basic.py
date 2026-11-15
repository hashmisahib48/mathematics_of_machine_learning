# 8.5  What KFold hands out, and why the order of the rows matters.
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_score)

for k, (tr, te) in enumerate(KFold(5).split(np.arange(10))):
    print("fold", k + 1, " test rows", te, " train rows", tr)
n = len(load_diabetes().target)
sizes = [len(te) for _, te in KFold(5).split(np.zeros(n))]
print("diabetes, n =", n, " fold sizes", sizes)

X, y = load_iris(return_X_y=True)      # rows are sorted by class
clf = LogisticRegression(max_iter=1000)
for name, cv in (
        ("KFold(3), no shuffle ", KFold(3)),
        ("KFold(3), shuffled   ",
         KFold(3, shuffle=True, random_state=0)),
        ("StratifiedKFold(3)   ", StratifiedKFold(3))):
    s = cross_val_score(clf, X, y, cv=cv)
    print(name, "accuracy per fold", np.round(s, 3))
