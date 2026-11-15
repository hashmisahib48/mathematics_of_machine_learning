# Problem 10B.1: ridge weights for many alpha, one SVD (Eq. 10.4-3)
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
Xc, yc = X - X.mean(axis=0), y - y.mean()      # centre: Eq. 10.3-3
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # the only SVD
z = U.T @ yc                                   # u_j' y_c, once


def ridge_path(alphas):
    """Weights for every alpha, Eq. 10.4-3."""
    return np.array([Vt.T @ (s / (s**2 + a) * z) for a in alphas])


alphas = np.logspace(-3, 1, 20)
W = ridge_path(alphas)
W_sk = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
diff = np.abs(W - W_sk).max()
print("largest difference from Ridge: %.2e" % diff)
print("singular values:\n", np.round(s, 3))

names = load_diabetes().feature_names
j = [names.index(k) for k in ("bmi", "s1", "s2")]
print("\n   alpha     df      bmi       s1       s2")
for k in range(0, 20, 3):
    f = s**2 / (s**2 + alphas[k])        # filter factors, Eq. 10.4-4
    w = W[k, j]
    print(f"{alphas[k]:8.4f} {f.sum():6.2f} "
          f"{w[0]:8.1f} {w[1]:8.1f} {w[2]:8.1f}")

a = alphas[9]                                  # one value of alpha
f = s**2 / (s**2 + a)
print(f"\nfilter factors at alpha = {a:.4f}: strongest {f[0]:.3f},"
      f" weakest {f[-1]:.3f}")
print("squares of the smallest singular values:", np.round(s[-2:]**2, 4))
