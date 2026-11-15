# Newton's method (IRLS) for the pass/fail data of section 16.1.
import numpy as np
from sklearn.linear_model import LogisticRegression

hours = np.arange(1.0, 9.0)                    # x = 1, ..., 8
passed = np.array([0, 0, 0, 1, 0, 1, 1, 1.0])  # y: 1 = passed
X = np.c_[np.ones(8), hours]                   # ones, then x
w = np.zeros(2)                                # (b, w) = (0, 0)

print(" t       b        w        J      |g|")
for t in range(6):
    z = X @ w                                  # scores
    p = 1 / (1 + np.exp(-z))                   # Eq. 16.3-1
    q = passed * np.log(p) + (1 - passed) * np.log(1 - p)
    J = -np.sum(q)                             # Eq. 16.3-5
    g = X.T @ (p - passed)                     # Eq. 16.3-7
    S = p * (1 - p)                            # diagonal of S
    H = X.T @ (X * S[:, None])                 # Eq. 16.3-8
    print(f"{t:2d} {w[0]:8.4f} {w[1]:8.4f} {J:8.4f} "
          f"{np.linalg.norm(g):8.1e}")
    if t == 0:
        print("g =", g, "  H =", H.tolist())
    w = w - np.linalg.solve(H, g)              # Eq. 16.3-10

ref = LogisticRegression(C=np.inf, tol=1e-10, max_iter=500)
ref.fit(hours[:, None], passed)
print("sklearn:", np.round(ref.intercept_, 4),
      np.round(ref.coef_.ravel(), 4))
print("odds ratio e^w =", round(float(np.exp(w[1])), 3))
p45 = 1 / (1 + np.exp(-(w[0] + 4.5 * w[1])))
print("P(pass | 4.5 h) =", round(float(p45), 4))
