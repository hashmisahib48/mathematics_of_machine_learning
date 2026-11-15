# Validation curves of AdaBoost on the two moons (five folds).
import json
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.33, random_state=0, stratify=y)
folds = list(StratifiedKFold(5, shuffle=True, random_state=0)
             .split(Xtr, ytr))
res = {"rounds": 300, "eta": {}, "depth": {}}
for eta in (1.0, 0.3, 0.1):
    acc = []
    for a, b in folds:
        m = AdaBoostClassifier(n_estimators=300, learning_rate=eta,
                               random_state=0).fit(Xtr[a], ytr[a])
        acc.append(list(m.staged_score(Xtr[b], ytr[b])))
    res["eta"][str(eta)] = np.mean(acc, axis=0).tolist()
    v = res["eta"][str(eta)]
    k = int(np.argmax(v)) + 1
    print(f"eta {eta:3}: best {max(v):.4f} at {k:3d} rounds;",
          "at 10, 50, 300:", *(f"{v[j]:.3f}" for j in (9, 49, 299)))
grid = [1, 2, 3, 4, 6, 8]
for d in grid:
    r = cross_validate(AdaBoostClassifier(
        DTC(max_depth=d), n_estimators=50, random_state=0),
        Xtr, ytr, cv=folds, return_train_score=True)
    tr, va = r["train_score"], r["test_score"]
    res["depth"][str(d)] = [tr.mean(), va.mean(), va.std()]
    print(f"base depth {d}: train {tr.mean():.3f}"
          f"  validation {va.mean():.3f}")
json.dump(res, open("../verify/choose_ada.json", "w"))
