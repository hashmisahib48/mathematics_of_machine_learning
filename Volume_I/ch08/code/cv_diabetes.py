# 8.5  cross_val_score and cross_validate on the diabetes data.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (KFold, cross_val_score,
                                     cross_validate)

X, y = load_diabetes(return_X_y=True)
ols = LinearRegression()
s = cross_val_score(ols, X, y, cv=5)       # default score: R2
print("5-fold R2   ", np.round(s, 3))
print(f"mean {s.mean():.3f}   sd {s.std(ddof=1):.3f}"
      f"   standard error {s.std(ddof=1) / np.sqrt(5):.3f}")
for seed in range(3):
    cv = KFold(5, shuffle=True, random_state=seed)
    s = cross_val_score(ols, X, y, cv=cv)
    print(f"shuffle, seed {seed}: mean R2 {s.mean():.3f}")

res = cross_validate(ols, X, y, cv=5, return_train_score=True)
print(sorted(res))
print("train R2", np.round(res["train_score"], 3))
print("test  R2", np.round(res["test_score"], 3))
base = cross_val_score(DummyRegressor(), X, y, cv=5)
print("mean baseline, R2 per fold", np.round(base, 3))
