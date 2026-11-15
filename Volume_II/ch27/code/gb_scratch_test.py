# Tests of the two scratch boosters against scikit-learn.
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.model_selection import train_test_split
from gb_scratch import MyGBR, MyGBC

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)
kw = dict(learning_rate=0.1, max_depth=3, random_state=0)
mine = MyGBR(n_estimators=100, **kw).fit(Xtr, ytr)
ref = GradientBoostingRegressor(n_estimators=100, **kw)
ref.fit(Xtr, ytr)
gap = np.abs(mine.predict(Xte) - ref.predict(Xte)).max()
print("regressor, 100 trees:  max |difference| =", gap)
print("  test R^2", round(ref.score(Xte, yte), 4))
assert gap < 1e-8

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
for depth, n in ((1, 100), (3, 10), (3, 100)):
    kw = dict(learning_rate=0.1, max_depth=depth, random_state=0)
    mine = MyGBC(n_estimators=n, **kw).fit(Xtr, ytr)
    ref = GradientBoostingClassifier(n_estimators=n, **kw)
    ref.fit(Xtr, ytr)
    gap = np.abs(mine.decision_function(Xte)
                 - ref.decision_function(Xte)).max()
    acc = ((mine.predict(Xte) == yte).mean(), ref.score(Xte, yte))
    print(f"classifier, depth {depth}, {n:3d} trees: max |diff| "
          f"{gap:.1e}, accuracy {acc[0]:.4f} and {acc[1]:.4f}")
    if depth == 1 or n == 10:
        assert gap < 1e-8
