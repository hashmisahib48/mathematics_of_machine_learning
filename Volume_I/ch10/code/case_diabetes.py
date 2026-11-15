# Case study: the diabetes data again, now with ridge.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

data = load_diabetes()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)       # split of Chapter 9

alphas = np.logspace(-3, 1, 41)                 # 0.001 ... 10
ols = LinearRegression().fit(X_tr, y_tr)
rcv = RidgeCV(alphas=alphas).fit(X_tr, y_tr)    # leave-one-out
print(f"alpha_ chosen by leave-one-out: {rcv.alpha_:.4f}")

s = np.linalg.svd(X_tr - X_tr.mean(axis=0), compute_uv=False)
df = np.sum(s**2 / (s**2 + rcv.alpha_))         # Eq. 10.4-6
print(f"effective degrees of freedom: {df:.2f} of {X.shape[1]}")

for name, m in (("least squares", ols), ("ridge (RidgeCV)", rcv)):
    rmse = np.sqrt(mean_squared_error(y_te, m.predict(X_te)))
    print(f"{name:16s} train R2 = {m.score(X_tr, y_tr):.3f}"
          f"  test R2 = {m.score(X_te, y_te):.3f}"
          f"  RMSE = {rmse:.1f}")

for name, est in (("least squares", LinearRegression()),
                  ("ridge", RidgeCV(alphas=alphas))):
    cv = cross_val_score(est, X, y, cv=5, scoring="r2")
    print(f"{name:14s} 5-fold R2 mean = {cv.mean():.3f}"
          f"  min = {cv.min():.3f}  max = {cv.max():.3f}")

print("\n         OLS   ridge")
for n, a, b in zip(data.feature_names, ols.coef_, rcv.coef_):
    print(f"  {n:4s} {a:7.1f} {b:7.1f}")
