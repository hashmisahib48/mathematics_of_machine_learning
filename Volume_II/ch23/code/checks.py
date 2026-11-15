# Checks of statements made in the text of Chapter 23.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, NuSVR, LinearSVR

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])


def lin(Xa, ya, **kw):
    return SVR(kernel="linear", tol=1e-10, **kw).fit(Xa, ya)


# 1. scale of y: SVR(s y; C, s eps) = s SVR(y; C / s, eps)   (Eq. 23.8-1)
s = 7.0
a = lin(X, s * y, C=2.0, epsilon=s * 0.5)
b = lin(X, y, C=2.0 / s, epsilon=0.5)
print("scale of y: max |f(s y) - s f(y)| =",
      np.abs(a.predict(X) - s * b.predict(X)).max())
# 2. dropping a sample inside the tube changes nothing
full = lin(X, y, C=1.0, epsilon=1.0)
keep = [1, 2, 3, 4]
no1 = lin(X[keep], y[keep], C=1.0, epsilon=1.0)
print("drop sample 1 (inside): w, b =", no1.coef_[0, 0].round(4),
      no1.intercept_[0].round(4))
for d in (1, 4):
    keep = [i for i in range(5) if i != d]
    m = lin(X[keep], y[keep], C=1.0, epsilon=1.0)
    print(f"drop sample {d + 1} (support vector): w, b =",
          m.coef_[0, 0].round(4), m.intercept_[0].round(4))
# 3. sample_weight multiplies C (the bound of beta_i)
sw = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
m = lin(X, y, C=0.1, epsilon=1.0)
m3 = SVR(kernel="linear", tol=1e-10, C=0.1, epsilon=1.0)
m3.fit(X, y, sample_weight=sw)
print("C = 0.1, dual_coef_ without weights:", m.dual_coef_.round(4),
      " with weight 3 on sample 5:", m3.dual_coef_.round(4),
      "support_", m3.support_)
# 4. epsilon = 0 is least absolute deviations
m0 = lin(X, y, C=1e4, epsilon=0.0)
q = QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
sad = lambda w, b0: np.abs(y - w * X[:, 0] - b0).sum()
print("eps = 0: SVR w, b, sum|r| =", m0.coef_[0, 0].round(4),
      m0.intercept_[0].round(4),
      sad(m0.coef_[0, 0], m0.intercept_[0]).round(4),
      " QuantileRegressor", q.coef_.round(4), q.intercept_.round(4),
      sad(q.coef_[0], q.intercept_).round(4))
# 5. the density of Section 23.10
C, eps = 2.0, 0.5
r = np.linspace(-40, 40, 800001)
dens = C / (2 * (1 + C * eps)) * np.exp(-C * np.maximum(0, abs(r) - eps))
print("density integral:", round(float(np.trapezoid(dens, r)), 6),
      " flat height", round(C / (2 * (1 + C * eps)), 4))
# 6. raw diabetes target: SVR(C=1, eps=0.1) = scaled-target SVR
Xd, yd = load_diabetes(return_X_y=True)
Xt, Xe, yt, ye = train_test_split(Xd, yd, test_size=0.25, random_state=0)
mu, sd = yt.mean(), yt.std()
raw = SVR(C=1.0, epsilon=0.1, tol=1e-8).fit(Xt, yt)
std = SVR(C=1.0 / sd, epsilon=0.1 / sd, tol=1e-8)
std.fit(Xt, (yt - mu) / sd)
diff = np.abs(raw.predict(Xe) - (mu + sd * std.predict(Xe))).max()
print(f"diabetes y: sd {sd:.2f}; raw SVR(C=1) test R2 "
      f"{raw.score(Xe, ye):.3f}")
print(f"  equals C = {1 / sd:.4f}, eps = {0.1 / sd:.4f} on the "
      f"standardised y: max diff {diff:.1e}")
# 7. nu limits
for nu in (0.0, 1.0, 1.5):
    try:
        NuSVR(nu=nu).fit(X, y)
        print("NuSVR nu =", nu, "accepted")
    except ValueError as e:
        print("NuSVR nu =", nu, "->", str(e)[:70])
# 8. LinearSVR with the default loss has no primal solver
try:
    LinearSVR(dual=False).fit(X, y)
except ValueError as e:
    print("LinearSVR(dual=False):", str(e)[:75])
# 9. SGDRegressor with the epsilon-insensitive loss and alpha = 1/(C n)
from sklearn.linear_model import SGDRegressor
rg = np.random.default_rng(0)
Xs = rg.normal(size=(3000, 3))
ys = Xs @ [2.0, -1.0, 0.5] + 3.0 + 0.3 * rg.normal(size=3000)
sv = SVR(kernel="linear", C=1.0, epsilon=0.3, tol=1e-6).fit(Xs, ys)
sg = SGDRegressor(loss="epsilon_insensitive", epsilon=0.3, alpha=1 / 3000,
                  penalty="l2", average=True, max_iter=2000, tol=None,
                  random_state=0).fit(Xs, ys)
print("SVR coef", np.round(sv.coef_[0], 4), round(sv.intercept_[0], 4))
print("SGD coef", np.round(sg.coef_, 4), round(sg.intercept_[0], 4))
# 10. variances of the noise densities of Section 23.10 (C = 2)
rr = np.linspace(-40, 40, 400001)
pl = 2.0 / 2 * np.exp(-2.0 * np.abs(rr))
print("Laplace variance C = 2:", round(float(np.trapezoid(pl * rr**2, rr)), 4))
