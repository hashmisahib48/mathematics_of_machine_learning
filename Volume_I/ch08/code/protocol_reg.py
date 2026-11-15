# 8.7  Split once, cross-validate on the training part, test once.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

d = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(
    d.data, d.target, test_size=0.25, random_state=0)   # step 1

ix = lambda *names: [d.feature_names.index(n) for n in names]
cands = {                      # name: (model, columns used)
    "mean baseline": (DummyRegressor(), ix("bmi")),
    "bmi only": (LinearRegression(), ix("bmi")),
    "bmi, bp, s5": (LinearRegression(), ix("bmi", "bp", "s5")),
    "all ten": (LinearRegression(), list(range(10)))}
cv_r2 = {}
for name, (m, cols) in cands.items():   # step 2: training part only
    s = cross_val_score(m, X_tr[:, cols], y_tr, cv=5, scoring="r2")
    cv_r2[name] = s.mean()
    print(f"{name:14s} CV R2 = {s.mean():6.3f}"
          f"  (sd {s.std(ddof=1):.3f})")

best = max(cv_r2, key=cv_r2.get)
m, cols = cands[best]
m.fit(X_tr[:, cols], y_tr)             # step 3: refit on training
p = m.predict(X_te[:, cols])           # step 4: test set, once
print("chosen:", best)
print(f"test R2 = {r2_score(y_te, p):.3f}   "
      f"RMSE = {root_mean_squared_error(y_te, p):.1f}")
