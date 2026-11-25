# A degree-9 polynomial on 12 points, with and without ridge.
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error as rmse

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 4, 12))
y = np.sin(1.3 * x) + 0.3 * x + rng.normal(0, 0.25, 12)
xt = np.sort(rng.uniform(x.min(), x.max(), 200))     # new data
yt = np.sin(1.3 * xt) + 0.3 * xt + rng.normal(0, 0.25, 200)
for est in (LinearRegression(), Ridge(0.001), Ridge(0.1), Ridge(10)):
    m = make_pipeline(PolynomialFeatures(9, include_bias=False),
                      StandardScaler(), est).fit(x[:, None], y)
    e1 = rmse(y, m.predict(x[:, None]))
    e2 = rmse(yt, m.predict(xt[:, None]))
    big = np.abs(m[-1].coef_).max()
    print(f"{str(est):18s} train {e1:.3f}  test {e2:.3f}"
          f"  max|w| {big:7.1f}")
