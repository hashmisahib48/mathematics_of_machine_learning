# Case study: diabetes with 10 % of the training targets wrecked.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (HuberRegressor, LinearRegression,
    QuantileRegressor, RANSACRegressor, TheilSenRegressor)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)

rng = np.random.default_rng(0)
bad = rng.choice(len(y_tr), size=33, replace=False)
y_bad = y_tr.copy()
y_bad[bad] += 300                      # 33 of 331 targets shifted

models = {
    "OLS": LinearRegression(),
    "Huber": HuberRegressor(max_iter=1000),
    "RANSAC": RANSACRegressor(min_samples=60,
        residual_threshold=100, max_trials=200, random_state=0),
    "Theil-Sen": TheilSenRegressor(random_state=0),
    "Quantile": QuantileRegressor(alpha=0.0),
}
clean = LinearRegression().fit(X_tr, y_tr)
ref = clean.predict(X_te)
rmse = np.sqrt(np.mean((y_tr - clean.predict(X_tr)) ** 2))
print(f"clean-fit training RMSE = {rmse:.1f}")
print("trained on     R2 (test)   MAE (test)")
print(f"OLS, clean y   {r2_score(y_te, ref):8.3f}"
      f"   {mean_absolute_error(y_te, ref):8.1f}")
for name, m in models.items():
    p = m.fit(X_tr, y_bad).predict(X_te)
    print(f"{name:10s}   {r2_score(y_te, p):8.3f}"
          f"   {mean_absolute_error(y_te, p):8.1f}")

dflt = RANSACRegressor(random_state=0).fit(X_tr, y_bad)
print("RANSAC defaults: R2 =",
      round(r2_score(y_te, dflt.predict(X_te)), 3))
h, r, t = models["Huber"], models["RANSAC"], models["Theil-Sen"]
print("Huber scale_ =", round(h.scale_, 1))
print("Huber flagged", h.outliers_.sum(), "samples;",
      h.outliers_[bad].sum(), "of them wrecked")
print("RANSAC kept", r.inlier_mask_.sum(), "samples;",
      r.inlier_mask_[bad].sum(), "of them wrecked")
print("Theil-Sen breakdown_ =", round(t.breakdown_, 3),
      "-> tolerates", int(t.breakdown_ * len(y_tr)), "bad targets")
