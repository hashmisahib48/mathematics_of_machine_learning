# Problem 9C.3: units and the size of weights
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
np.set_printoptions(precision=4, suppress=True)
x = np.array([1, 2, 3, 4, 5.]); y = np.array([3, 4, 6, 7, 10.])
for name, f in (("km", 1), ("m", 1000)):
    m = LinearRegression().fit(f * x[:, None], y)
    print(f"{name:2s}: w = {m.coef_[0]:.4f} min per unit, b = {m.intercept_:.4f}, R2 = {m.score(f * x[:, None], y):.4f},"
          f" predictions agree: {np.allclose(m.predict(f * x[:, None]), 0.9 + 1.7 * x)}")
print("w(m) * 1000 =", round(1000 * LinearRegression().fit(1000 * x[:, None], y).coef_[0], 4))

print("\nDiabetes data in natural units (load_diabetes(scaled=False)): weights and weights x standard deviation")
d = load_diabetes(scaled=False); X, t = d.data, d.target
m = LinearRegression().fit(X, t)
sd = X.std(axis=0)
print(f"{'':4s} {'std':>8s} {'w (raw)':>9s} {'w * std':>9s}")
for n, s, w in zip(d.feature_names, sd, m.coef_):
    print(f"{n:4s} {s:8.2f} {w:9.2f} {w * s:9.2f}")
rk_raw = np.argsort(-np.abs(m.coef_)); rk_std = np.argsort(-np.abs(m.coef_ * sd))
print("order by |w|      :", [d.feature_names[i] for i in rk_raw])
print("order by |w * std|:", [d.feature_names[i] for i in rk_std])
Xs = (X - X.mean(0)) / sd
ms = LinearRegression().fit(Xs, t); print("standardised-fit weights equal w * std:", np.allclose(ms.coef_, m.coef_ * sd))
print("predictions identical to the scaled data set of the book:", np.allclose(m.predict(X), LinearRegression().fit(load_diabetes().data, t).predict(load_diabetes().data)))
