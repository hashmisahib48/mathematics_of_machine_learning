# Gradient descent on the five students and on the deliveries.
from gd import LinearRegressionGD, sgd

x = [[1], [2], [3], [4], [5]]
y = [52, 57, 61, 68, 72]
for eta in (0.1, 0.25, 0.45, 0.55):
    m = LinearRegressionGD(eta=eta, n_iter=200).fit(x, y)
    print(f"eta {eta:4.2f}: {m.n_iter_:3d} iterations,"
          f" w = {m.coef_[0]:.4g}  b = {m.intercept_:.4g}")
w, b = sgd(x, y, eta0=0.05, epochs=50)
print("SGD, 50 epochs: w =", round(w[0], 4), " b =", round(b, 4))
X = [[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]]          # deliveries
t = [14, 23, 24, 35, 38]
m = LinearRegressionGD(eta=0.15, n_iter=5000).fit(X, t)
print("deliveries:", m.n_iter_, "iterations, w =",
      [round(v, 4) for v in m.coef_], " b =", round(m.intercept_, 3))
