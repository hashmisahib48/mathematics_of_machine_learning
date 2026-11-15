# Case study, continued: coefficients and cross-validation.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (ElasticNetCV, LassoCV,
                                  LinearRegression)
from sklearn.model_selection import (cross_val_score,
                                     train_test_split)

data = load_diabetes()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)

fits = {"OLS": LinearRegression(),
        "LassoCV": LassoCV(cv=5, random_state=0),
        "ElasticNetCV 0.5": ElasticNetCV(l1_ratio=0.5, eps=1e-4,
                                         cv=5, random_state=0)}
for m in fits.values():
    m.fit(X_tr, y_tr)

print("          " + "".join(f"{k:>18s}" for k in fits))
for j, name in enumerate(data.feature_names):
    if name in ("s1", "s2"):
        print(f"  {name:5s}  " + "".join(
            f"{m.coef_[j] + 0.0:18.1f}" for m in fits.values()))

print("\nnested 5-fold R2, alpha chosen inside each fold")
for name, m in fits.items():
    s = cross_val_score(m, X, y, cv=5, scoring="r2")
    print(f"  {name:17s} mean {s.mean():.3f} ", np.round(s, 2))

short = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0)  # eps=1e-3
s = cross_val_score(short, X, y, cv=5, scoring="r2")
print("  default eps       mean", f"{s.mean():.3f}",
      np.round(s, 2))
v = (X_tr ** 2).mean(axis=0)
print("mean squares of X_tr columns:", round(v.min(), 4),
      "to", round(v.max(), 4))
