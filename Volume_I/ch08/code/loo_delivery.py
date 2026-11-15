# 8.5  Five-fold cross-validation of the delivery data, by hand
#      (five folds of five samples = one sample per fold).
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

x = np.array([1., 2, 3, 4, 5])
y = np.array([3., 4, 6, 7, 10])
print(" k  held-out x    w      b   line error  mean error")
e_line, e_mean = [], []
for k, (tr, te) in enumerate(KFold(5).split(x), start=1):
    xm, ym = x[tr].mean(), y[tr].mean()
    dx = x[tr] - xm
    w = np.sum(dx * (y[tr] - ym)) / np.sum(dx**2)
    b = ym - w * xm                    # Eq. 9.3-5 on four samples
    e_line.append(y[te][0] - (b + w * x[te][0]))
    e_mean.append(y[te][0] - ym)
    print(f"{k:2d} {x[te][0]:8.0f} {w:8.3f} {b:6.3f}"
          f"{e_line[-1]:9.3f} {e_mean[-1]:11.3f}")
print("CV MSE, line:", round(np.mean(np.square(e_line)), 4),
      "  mean:", round(np.mean(np.square(e_mean)), 4))
fit = LinearRegression().fit(x.reshape(-1, 1), y)
res = y - fit.predict(x.reshape(-1, 1))
print("training MSE of the line:", round(np.mean(res**2), 4))
X = x.reshape(-1, 1)
for m in (LinearRegression(), DummyRegressor()):
    s = cross_val_score(m, X, y, cv=5,
                        scoring="neg_mean_squared_error")
    print(type(m).__name__, "-> mean of scores", round(s.mean(), 4))
