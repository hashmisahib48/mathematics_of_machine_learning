# Separable data: the unpenalised loss has no minimum.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x = np.arange(1.0, 9.0)
y = np.array([0, 0, 0, 1, 1, 1, 1, 1])         # student 5 passes
m = 2.0 * y - 1                                # labels -1, +1

print("scale s    w     J(s)   (boundary fixed at 3.5 h)")
for s in (1, 2, 4, 8, 16):
    z = s * (x - 3.5)                          # w = s, b = -3.5 s
    print(f"{s:5d} {s:6.1f} {np.logaddexp(0, -m * z).sum():9.6f}")

print("\n    C     w       b     boundary")
for C in (1, 10, 100, 1000, 1e6):
    lr = LogisticRegression(C=C, tol=1e-10, max_iter=10000)
    lr.fit(x[:, None], y)
    w, b = lr.coef_[0, 0], lr.intercept_[0]
    print(f"{C:7.0e} {w:7.3f} {b:8.3f} {-b / w:8.3f}")

X, t = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
free = LogisticRegression(C=np.inf, max_iter=5000).fit(X, t)
norm = np.linalg.norm(free.coef_)
print("\nbreast cancer, no penalty: training accuracy",
      free.score(X, t), f" |w| = {norm:.1e}")
