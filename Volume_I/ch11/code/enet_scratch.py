import numpy as np
from sklearn.linear_model import ElasticNet, Lasso

def soft(z, a):                                  # Eq. 11.3-6
    return np.sign(z) * np.maximum(np.abs(z) - a, 0.0)

class MyElasticNet:                              # Eq. 11.4-1
    def __init__(self, alpha=1.0, l1_ratio=1.0, tol=1e-12):
        self.alpha, self.l1_ratio, self.tol = alpha, l1_ratio, tol

    def fit(self, X, y, max_iter=10000):
        n, p = X.shape
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym                  # Eq. 11.3-2
        v = (Xc ** 2).mean(axis=0)               # Eq. 11.3-3
        a = self.alpha * self.l1_ratio           # lambda_1
        b = self.alpha * (1 - self.l1_ratio)     # lambda_2
        w, r = np.zeros(p), yc.copy()            # r = y_c - X_c w
        for sweep in range(1, max_iter + 1):
            biggest = 0.0
            for j in range(p):
                z = Xc[:, j] @ r / n + v[j] * w[j]   # Eq. 11.3-10
                w_new = soft(z, a) / (v[j] + b)      # Eq. 11.4-3
                r += Xc[:, j] * (w[j] - w_new)
                biggest = max(biggest, abs(w_new - w[j]))
                w[j] = w_new
            if biggest < self.tol:
                break
        self.coef_, self.n_iter_ = w, sweep
        self.intercept_ = ym - xm @ w            # Eq. 11.3-2
        return self

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 8))
X[:, 1] = X[:, 0] + 0.3 * rng.normal(size=60)    # correlated pair
y = X @ [3.0, 0, 0, -2.0, 0, 0, 1.0, 0] + 5 + rng.normal(size=60)
for rho in (1.0, 0.5):
    kw = dict(alpha=0.3, tol=1e-12, max_iter=100000)
    ref = (Lasso(**kw) if rho == 1 else
           ElasticNet(l1_ratio=rho, **kw)).fit(X, y)
    me = MyElasticNet(0.3, rho).fit(X, y)
    assert np.allclose(me.coef_, ref.coef_, atol=1e-8)
    assert abs(me.intercept_ - ref.intercept_) < 1e-8
    print(f"rho={rho}: {me.n_iter_} sweeps, w =")
    print("  ", np.round(me.coef_, 3) + 0.0)
