# Problem 23B.2: sparsity against accuracy on a noisy sine.
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

rng = np.random.default_rng(1)
x = rng.uniform(0, 6, 300)
noise = 0.25 * rng.normal(size=300)
y = np.sin(x) + noise
X = x[:, None]
X_tr, y_tr, X_te, y_te = X[:200], y[:200], X[200:], y[200:]
rmse = lambda p, t: np.sqrt(np.mean((p - t) ** 2))

print("eps    SV  share  RMSE(y)  RMSE(sin x)")
for eps in (0, 0.1, 0.2, 0.3, 0.5, 0.8):
    m = SVR(C=10, gamma=0.5, epsilon=eps).fit(X_tr, y_tr)
    p = m.predict(X_te)
    nsv = len(m.support_)
    t = np.sin(X_te[:, 0])                  # the true curve
    ry = rmse(p, y_te)
    print(f"{eps:3.1f} {nsv:5d} {nsv / 200:6.3f} {ry:8.3f}"
          f" {rmse(p, t):10.3f}")
# kernel ridge has no intercept: centre y first (Chapter 21)
k = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
k.fit(X_tr, y_tr - y_tr.mean())
p = k.predict(X_te) + y_tr.mean()
print(f"KRR   200  1.000 {rmse(p, y_te):8.3f}"
      f" {rmse(p, t):10.3f}")
print("noise of the test targets:", round(noise[200:].std(), 3),
      " (true sd 0.25)")
