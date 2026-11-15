# NuSVR: nu bounds the points outside the tube and the SVs.
import numpy as np
from sklearn.svm import SVR, NuSVR

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 6, 200))[:, None]
y = np.sin(X[:, 0]) + 0.25 * rng.normal(size=200)
C, gamma = 10.0, 0.5

print(" nu  outside  SV share  tube eps  sum|b|/Cn  |f - SVR|")
for nu in (0.1, 0.3, 0.5, 0.7, 0.9):
    m = NuSVR(nu=nu, C=C, gamma=gamma, tol=1e-8).fit(X, y)
    r = y - m.predict(X)
    d = np.abs(m.dual_coef_[0])
    free = (d > 1e-6) & (d < C - 1e-6)
    eps = np.abs(r[m.support_[free]]).mean()   # tube found
    out = np.mean(np.abs(r) > eps + 1e-6)
    share = len(m.support_) / len(y)
    ref = SVR(C=C, epsilon=eps, gamma=gamma, tol=1e-8).fit(X, y)
    diff = np.abs(ref.predict(X) - m.predict(X)).max()
    assert out <= nu <= share
    s1 = d.sum() / (C * len(y))                # equals nu
    print(f"{nu:4.1f} {out:7.3f} {share:9.3f} {eps:9.4f} {s1:9.3f}"
          f" {diff:10.1e}")
