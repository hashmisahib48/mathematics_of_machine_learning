# The error messages and shapes quoted in Chapter 27.
import warnings
import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)

X, y = load_iris(return_X_y=True)
Xn = X.copy()
Xn[0, 0] = np.nan
for label, make, data in (
        ("NaN in X", lambda: GradientBoostingClassifier(), Xn),
        ("exponential, 3 classes",
         lambda: GradientBoostingClassifier(loss="exponential"), X),
        ("loss='deviance'",
         lambda: GradientBoostingClassifier(loss="deviance"), X)):
    try:
        make().fit(data, y)
    except Exception as e:
        text = str(e).replace("\n", " ").split(". ")
        print(f"{label}: {type(e).__name__}: {text[0]}. {text[1][:60]}")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    GradientBoostingRegressor(criterion="friedman_mse",
                              n_estimators=2).fit(X, y)
print("criterion given:", w[0].category.__name__)

Xc, yc = load_breast_cancer(return_X_y=True)
g = GradientBoostingClassifier(n_estimators=3).fit(Xc, yc)
F = next(g.staged_decision_function(Xc))
print("decision_function:", g.decision_function(Xc).shape,
      " one stage of staged_decision_function:", F.shape)
print("yc - expit(F):", (yc - expit(F)).shape,
      " with F.ravel():", (yc - expit(F.ravel())).shape)
