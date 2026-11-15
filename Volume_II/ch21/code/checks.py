# Numbers quoted in the prose of Chapter 21 (not printed as a listing).
import time
import numpy as np
from math import factorial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import (additive_chi2_kernel, chi2_kernel,
                                      cosine_similarity, laplacian_kernel,
                                      pairwise_kernels, rbf_kernel,
                                      sigmoid_kernel)

rng = np.random.default_rng(0)
# 1. positive semi-definiteness
X = rng.normal(size=(8, 3))
for name, K in (("linear", X @ X.T), ("rbf", rbf_kernel(X, gamma=0.5)),
                ("laplacian", laplacian_kernel(X, gamma=0.5)),
                ("sigmoid", sigmoid_kernel(X, gamma=1.0, coef0=0.0))):
    print(f"min eigenvalue {name:9s}: {np.linalg.eigvalsh(K).min():.4f}")
K1, K2 = rbf_kernel(X, gamma=0.5), X @ X.T
print("sum  min eig:", round(np.linalg.eigvalsh(K1 + K2).min(), 4))
print("prod min eig:", round(np.linalg.eigvalsh(K1 * K2).min(), 4))
Xt = np.array([[-1.0], [0.0], [1.0]])
print("toy rbf eigenvalues:",
      np.round(np.linalg.eigvalsh(rbf_kernel(Xt, gamma=0.5)), 4))
# 2. chi2 formulas
P = rng.uniform(0.1, 1, size=(3, 4))
ac = -(((P[0] - P[1]) ** 2) / (P[0] + P[1])).sum()
print("additive chi2 formula:", np.isclose(additive_chi2_kernel(P)[0, 1], ac),
      " chi2:", np.isclose(chi2_kernel(P, gamma=0.7)[0, 1], np.exp(0.7 * ac)))
c = P[0] @ P[1] / np.linalg.norm(P[0]) / np.linalg.norm(P[1])
print("cosine formula:", np.isclose(cosine_similarity(P)[0, 1], c))
print("pairwise_kernels names:", sorted(
    __import__("sklearn.metrics.pairwise", fromlist=["x"]
               ).PAIRWISE_KERNEL_FUNCTIONS))
# 3. truncated series for the rbf kernel, x=0.5, z=1.5, gamma=1
g, x, z = 1.0, 0.5, 1.5
for m in (1, 2, 3, 5, 8):
    s = np.exp(-g * x * x - g * z * z) * sum(
        (2 * g * x * z) ** j / factorial(j) for j in range(m))
    print(f"terms {m}: {s:.4f}")
print("exact:", round(float(np.exp(-g * (x - z) ** 2)), 4))
# 4. sample_weight: weight 2 equals duplicating the sample
Xs = rng.normal(size=(10, 2)); ys = rng.normal(size=10)
w = np.ones(10); w[3] = 2.0
a = KernelRidge(alpha=0.5, kernel="rbf").fit(Xs, ys, sample_weight=w)
b = KernelRidge(alpha=0.5, kernel="rbf").fit(
    np.vstack([Xs, Xs[3:4]]), np.r_[ys, ys[3]])
print("weight 2 = duplicate:",
      np.allclose(a.predict(Xs), b.predict(Xs)))
# 5. posterior mean of a Gaussian process equals kernel ridge, alpha = noise
ell = 0.8
gp = GaussianProcessRegressor(RBF(ell), alpha=0.25, optimizer=None)
kr = KernelRidge(alpha=0.25, kernel="rbf", gamma=1 / (2 * ell ** 2))
Xn = rng.normal(size=(5, 2))
print("GP mean = KRR:", np.allclose(gp.fit(Xs, ys).predict(Xn),
                                      kr.fit(Xs, ys).predict(Xn)))
# 6. cost: fit time against n
for n in (500, 1000, 2000, 4000):
    Xb = rng.normal(size=(n, 10)); yb = rng.normal(size=n)
    t = time.perf_counter()
    KernelRidge(alpha=1.0, kernel="rbf").fit(Xb, yb)
    print(f"n={n:5d}: fit {time.perf_counter() - t:6.2f} s,"
          f"  Gram matrix {8 * n * n / 1e6:7.1f} MB")
print("n=100000 Gram matrix:", 8 * 100000 ** 2 / 1e9, "GB")
# 7. chi2 kernels inside KernelRidge (non-negative features)
import warnings
Xp = rng.uniform(0.1, 1, size=(10, 3)); yp = rng.normal(size=10)
try:
    KernelRidge(kernel="chi2").fit(Xp, yp)
    print("chi2, gamma=None: fitted")
except Exception as err:
    print("chi2, gamma=None:", type(err).__name__)
KernelRidge(kernel="chi2", gamma=1.0).fit(Xp, yp)
print("chi2, gamma=1.0: fitted")
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    KernelRidge(kernel="additive_chi2").fit(Xp, yp)
print("additive_chi2 warnings:", [str(w_.message)[:40] for w_ in wl])
# 8. alpha as an array: one penalty per target
Y2 = np.c_[yp, 2 * yp]
m2 = KernelRidge(alpha=[0.1, 10.0], kernel="rbf").fit(Xp, Y2)
m_a = KernelRidge(alpha=0.1, kernel="rbf").fit(Xp, yp)
m_b = KernelRidge(alpha=10.0, kernel="rbf").fit(Xp, 2 * yp)
print("alpha array:", m2.dual_coef_.shape,
      np.allclose(m2.dual_coef_[:, 0], m_a.dual_coef_),
      np.allclose(m2.dual_coef_[:, 1], m_b.dual_coef_))
# 9. a callable and a Gaussian-process kernel object as `kernel`
from sklearn.gaussian_process.kernels import RBF as GPRBF
Xg = rng.normal(size=(12, 2)); yg = rng.normal(size=12)
ref = KernelRidge(alpha=0.3, kernel="rbf", gamma=0.5).fit(Xg, yg)
def bump(u, v, s):
    return float(np.exp(-s * np.sum((u - v) ** 2)))
c1 = KernelRidge(alpha=0.3, kernel=bump,
                 kernel_params={"s": 0.5}).fit(Xg, yg)
c2 = KernelRidge(alpha=0.3, kernel=GPRBF(1.0)).fit(Xg, yg)
print("callable with kernel_params:",
      np.allclose(c1.dual_coef_, ref.dual_coef_))
print("GP kernel RBF(length_scale=1):",
      np.allclose(c2.dual_coef_, ref.dual_coef_))
# 10. polynomial columns of Chapter 15 and make_friedman1
from math import comb
print("columns, degree 3, p=10:", comb(13, 3), " p=100:", comb(103, 3),
      " p=20:", comb(23, 3))
from sklearn.datasets import make_friedman1
Xq, yq = make_friedman1(500, noise=0.0, random_state=0)
tr = (10 * np.sin(np.pi * Xq[:, 0] * Xq[:, 1]) + 20 * (Xq[:, 2] - 0.5) ** 2
      + 10 * Xq[:, 3] + 5 * Xq[:, 4])
print("friedman1 formula:", np.allclose(yq, tr))
from sklearn.linear_model import LinearRegression
Xq, yq = make_friedman1(20000, noise=0.0, random_state=0)
print("best linear R2 on noise-free friedman1:",
      round(LinearRegression().fit(Xq, yq).score(Xq, yq), 3))
Xq, yq = make_friedman1(20000, noise=1.0, random_state=0)
print("best linear R2, noise 1:",
      round(LinearRegression().fit(Xq, yq).score(Xq, yq), 3))
# 11. alpha of the case study in the units of the shipped diabetes data
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
Xd, yd = load_diabetes(return_X_y=True)
Xd_tr, _, _, _ = train_test_split(Xd, yd, test_size=0.25, random_state=0)
print("n_train:", len(Xd_tr), " test:", len(Xd) - len(Xd_tr),
      " mean std:", round(Xd_tr.std(axis=0).mean(), 4),
      " X^T X factor:", round(1 / (Xd_tr.std(axis=0) ** 2).mean(), 0),
      " alpha 31.623 in old units:",
      round(31.623 * (Xd_tr.std(axis=0) ** 2).mean(), 3))
