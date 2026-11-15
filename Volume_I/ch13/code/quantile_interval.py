# A 90 % prediction band from quantile regression.
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss

rng = np.random.default_rng(0)


def make(n):
    x = rng.uniform(0, 10, n)
    y = 2 + 1.5 * x + rng.normal(size=n) * (0.3 + 0.35 * x)
    return x[:, None], y                    # noise grows with x


X_tr, y_tr = make(400)
X_te, y_te = make(4000)

for tau in (0.05, 0.5, 0.95):
    m = QuantileRegressor(quantile=tau, alpha=0.0).fit(X_tr, y_tr)
    p = m.predict(X_te)
    cover = np.mean(y_te <= p)              # share of targets below
    loss = mean_pinball_loss(y_te, p, alpha=tau)
    print(f"tau={tau:4.2f}  b={m.intercept_:6.3f}"
          f"  w={m.coef_[0]:6.3f}  below={cover:5.3f}"
          f"  pinball={loss:5.3f}")
