# Section 55.7: the two ways to lose the groups.
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, GroupKFold,
                                     cross_val_score)

rng = np.random.RandomState(0)
X, y = rng.normal(size=(40, 3)), np.tile([0, 1], 20)
g = np.repeat(np.arange(10), 4)             # 10 patients, 4 visits
lr = LogisticRegression()
try:
    cross_val_score(lr, X, y, cv=GroupKFold(5))
except ValueError as e:
    print("ValueError:", e)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cross_val_score(lr, X, y, groups=g, cv=KFold(5))
print("UserWarning:", w[0].message)
cv = KFold(5, shuffle=True, random_state=0)
shared = sum(np.isin(g[te], g[tr]).sum() for tr, te in cv.split(X))
print("held-out rows whose patient is also in training:",
      shared, "of", len(y))
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, cross_val_predict
try:
    cross_val_predict(Ridge(), X, y, cv=ShuffleSplit(3))
except ValueError as e:
    print("cross_val_predict:", str(e)[:48])
