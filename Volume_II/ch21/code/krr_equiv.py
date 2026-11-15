# Kernel ridge equals ridge on explicit columns (Section 21.4).
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

rng = np.random.default_rng(1)
X = rng.normal(size=(30, 2))
y = X[:, 0] * X[:, 1] + X[:, 0] ** 2 + 0.1 * rng.normal(size=30)
X_new = rng.normal(size=(4, 2))
alpha = 0.7


def phi(M):                         # Eq. 21.3-2, one row per sample
    a, b, s = M[:, 0], M[:, 1], np.sqrt(2)
    one = np.ones(len(M))
    return np.c_[one, s * a, s * b, a * a, s * a * b, b * b]


# 1. linear kernel = ridge without intercept
lin = KernelRidge(alpha=alpha, kernel="linear").fit(X, y)
rid = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
w = X.T @ lin.dual_coef_                       # w = X^T a
print("1. w = X^T a equals coef_:", np.allclose(w, rid.coef_))
print("   predictions equal     :",
      np.allclose(lin.predict(X_new), rid.predict(X_new)))

# 2. (x.z + 1)^2 kernel = ridge on the six columns phi(x)
pol = KernelRidge(alpha=alpha, kernel="poly", degree=2,
                  gamma=1.0, coef0=1.0).fit(X, y)
rid6 = Ridge(alpha=alpha, fit_intercept=False).fit(phi(X), y)
print("2. poly kernel = ridge on phi:",
      np.allclose(pol.predict(X_new), rid6.predict(phi(X_new))))
print("   weights, six columns:", np.round(rid6.coef_, 3))

# 3. centred data give ridge with an intercept
Xc, yc = X - X.mean(0), y - y.mean()
cen = KernelRidge(alpha=alpha, kernel="linear").fit(Xc, yc)
rid_b = Ridge(alpha=alpha).fit(X, y)
p = cen.predict(X_new - X.mean(0)) + y.mean()
print("3. centred KRR = Ridge(fit_intercept=True):",
      np.allclose(p, rid_b.predict(X_new)))
