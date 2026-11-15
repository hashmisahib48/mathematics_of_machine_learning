# Stacking by hand: six points, two level-0 models, three folds.
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

x = np.arange(1.0, 7.0)
y = np.array([2.0, 5.0, 3.0, 8.0, 7.0, 11.0])

def line(xs, ys):                       # Eq. 9.3-5 (Volume I)
    w = np.sum((xs - xs.mean()) * (ys - ys.mean())) / np.sum(
        (xs - xs.mean()) ** 2)
    return lambda t: ys.mean() + w * (t - xs.mean())

def nearest(xs, ys):                    # 1-nearest neighbour
    return lambda t: ys[np.argmin(np.abs(xs[:, None] - t), axis=0)]

oof = np.zeros((6, 2))                  # Eq. 29.4-1
for held in ([0, 1], [2, 3], [4, 5]):
    keep = np.setdiff1d(np.arange(6), held)
    for m, fit in enumerate((line, nearest)):
        oof[held, m] = fit(x[keep], y[keep])(x[held])
print("out-of-fold columns A (line), B (nearest):")
print(np.c_[x, y, oof])

G, c = oof.T @ oof, oof.T @ y       # normal eqs, no intercept
w = np.linalg.solve(G, c)
print("Z'Z =", G.round(2).tolist(), " Z'y =", c.round(2).tolist())
print("weights:", w.round(4))

new = np.array([4.4])                   # Eq. 29.4-3
z = np.array([line(x, y)(new)[0], nearest(x, y)(new)[0]])
print("level-0 on all six points:", z, " stacked:",
      round(z @ w, 4), " plain average:", round(z.mean(), 4))

sk = StackingRegressor(
    [("line", LinearRegression()),
     ("nearest", KNeighborsRegressor(n_neighbors=1))],
    final_estimator=LinearRegression(fit_intercept=False), cv=3)
sk.fit(x.reshape(-1, 1), y)
assert np.allclose(sk.final_estimator_.coef_, w)
assert np.allclose(sk.predict([[4.4]]), z @ w)
print("StackingRegressor agrees:",
      sk.final_estimator_.coef_.round(4))
