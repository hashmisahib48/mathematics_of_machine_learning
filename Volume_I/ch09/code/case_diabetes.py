# Case study: disease progression from ten baseline measurements.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

data = load_diabetes()
X, y = data.data, data.target
print("X:", X.shape, "  y:", y.shape)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)

base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
ols = LinearRegression().fit(X_tr, y_tr)

for name, m in (("mean baseline", base), ("least squares", ols)):
    p = m.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, p))
    print(f"{name:14s} test R2 = {r2_score(y_te, p):6.3f}"
          f"   RMSE = {rmse:5.1f}")

print(f"train R2 = {ols.score(X_tr, y_tr):.3f}")
cv = cross_val_score(LinearRegression(), X, y, cv=5,
                     scoring="r2")
print("5-fold R2 =", np.round(cv, 3))
print("mean      =", round(cv.mean(), 3))

print(f"intercept = {ols.intercept_:.1f}")
for name, w in zip(data.feature_names, ols.coef_):
    print(f"  {name:4s} {w:8.1f}")
