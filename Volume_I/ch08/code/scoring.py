# 8.6  The scoring argument and the sign convention.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import cross_val_score, cross_validate

X, y = load_diabetes(return_X_y=True)
ols = LinearRegression()
print(len(get_scorer_names()), "scoring names in this release")
for sc in ("r2", "neg_mean_squared_error",
           "neg_root_mean_squared_error",
           "neg_mean_absolute_error"):
    s = cross_val_score(ols, X, y, cv=5, scoring=sc)
    print(f"{sc:28s} mean {s.mean():9.3f}")

neg = cross_val_score(ols, X, y, cv=5,
                      scoring="neg_mean_squared_error")
print("MSE per fold, sign flipped:", np.round(-neg, 1))
print("sqrt of the mean MSE:", round(np.sqrt(-neg.mean()), 3))

loss = make_scorer(mean_squared_error, greater_is_better=False)
s2 = cross_val_score(ols, X, y, cv=5, scoring=loss)
print("make_scorer gives the same :", np.allclose(neg, s2))

res = cross_validate(ols, X, y, cv=5, scoring=["r2",
                     "neg_root_mean_squared_error"])
print(sorted(k for k in res if k.startswith("test_")))
