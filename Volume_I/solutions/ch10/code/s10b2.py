# Problem 10B.2: a noisy copy of bmi
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
rng = np.random.default_rng(0)
copy = X[:, 2] + 0.01 * X[:, 2].std() * rng.normal(size=len(X))
X2 = np.c_[X, copy]                            # 11 columns, copy last
Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.25,
                                      random_state=0)
r = np.corrcoef(X[:, 2], copy)[0, 1]
print("correlation of bmi and its copy: %.5f" % r)

ols = LinearRegression().fit(Xtr, ytr)
rdg = Ridge(alpha=0.05).fit(Xtr, ytr)
for name, m in (("least squares", ols), ("ridge, 0.05  ", rdg)):
    w = m.coef_[[2, 10]]
    print(f"{name}: bmi {w[0]:8.1f}  copy {w[1]:8.1f}"
          f"  sum {w.sum():7.1f}  test R2 {m.score(Xte, yte):.3f}")

Xc = Xtr - Xtr.mean(axis=0)
lam, V = np.linalg.eigh(Xc.T @ Xc)             # ascending eigenvalues
print("\nsmallest eigenvalue of Xc'Xc: %.2e" % lam[0])
print("its eigenvector on (bmi, copy):", np.round(V[[2, 10], 0], 3))
f = lam[0] / (lam[0] + 0.05)
print("filter factor at alpha = 0.05: %.4f" % f)
f2 = lam[1] / (lam[1] + 0.05)
print("second smallest: %.4f, factor %.3f" % (lam[1], f2))
