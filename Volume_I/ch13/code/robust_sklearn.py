# The four robust regressors on the delivery data with y5 = 20.
import numpy as np
from sklearn.linear_model import (HuberRegressor, LinearRegression,
    QuantileRegressor, RANSACRegressor, TheilSenRegressor)

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 20.0])

models = {
    "OLS": LinearRegression(),
    "Huber": HuberRegressor(),
    "RANSAC": RANSACRegressor(random_state=0),
    "Theil-Sen": TheilSenRegressor(random_state=0),
    "Quantile": QuantileRegressor(alpha=0.0),
}
print("model       b      w   predict(6)")
for name, m in models.items():
    m.fit(X, y)
    est = getattr(m, "estimator_", m)      # RANSAC: inner model
    b, w = est.intercept_, est.coef_[0]
    p6 = m.predict([[6.0]])[0]
    print(f"{name:10s} {b:6.3f} {w:6.3f} {p6:8.3f}")

hub, ran = models["Huber"], models["RANSAC"]
ts = models["Theil-Sen"]
print("Huber scale_    =", round(hub.scale_, 4))
print("Huber outliers_ =", hub.outliers_)
print("RANSAC inliers  =", ran.inlier_mask_)
print("RANSAC trials   =", ran.n_trials_)
print("Theil-Sen subsets =", ts.n_subpopulation_)
print("Theil-Sen breakdown_ =", round(ts.breakdown_, 3))

q = QuantileRegressor().fit(X, y)          # default alpha = 1.0
print("Quantile, default alpha:", q.coef_, q.intercept_)
