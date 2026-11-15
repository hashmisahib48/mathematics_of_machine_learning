# The other approximate feature maps, and two details of the source.
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.random_projection import GaussianRandomProjection
from sklearn.kernel_approximation import (AdditiveChi2Sampler,
    Nystroem, PolynomialCountSketch, RBFSampler, SkewedChi2Sampler)
from sklearn.metrics.pairwise import polynomial_kernel

X = load_digits(return_X_y=True)[0][:300] / 16   # in [0, 1]
rel = lambda K, Kh: np.linalg.norm(K - Kh) / np.linalg.norm(K)

# additive chi2: k = sum_j 2 x_j z_j / (x_j + z_j)
num = 2 * X[:, None, :] * X[None, :, :]
den = X[:, None, :] + X[None, :, :]
Kc = np.where(den > 0, num / np.where(den > 0, den, 1), 0).sum(2)
for s in (1, 2, 3):
    Z = AdditiveChi2Sampler(sample_steps=s).fit_transform(X)
    print(f"AdditiveChi2Sampler steps {s}: 64 -> {Z.shape[1]:3d} "
          f"features, relative Gram error {rel(Kc, Z @ Z.T):.4f}")

# skewed chi2: k = prod_j 2 sqrt((x_j+c)(z_j+c)) / (x_j+z_j+2c)
c = 0.5
A, B = X[:, None, :] + c, X[None, :, :] + c
Ks = np.prod(2 * np.sqrt(A * B) / (A + B), axis=2)
for D in (100, 1000, 10000):
    Z = SkewedChi2Sampler(skewedness=c, n_components=D,
                          random_state=0).fit_transform(X)
    print(f"SkewedChi2Sampler D = {D:5d}: relative Gram error "
          f"{rel(Ks, Z @ Z.T):.4f}")

K2 = polynomial_kernel(X, degree=2, gamma=1.0, coef0=0.0)
for D in (64, 640, 6400):
    Z = PolynomialCountSketch(degree=2, gamma=1.0, coef0=0.0,
                              n_components=D,
                              random_state=0).fit_transform(X)
    print(f"PolynomialCountSketch D = {D:4d}: relative Gram error "
          f"{rel(K2, Z @ Z.T):.4f}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    nys = Nystroem(n_components=500, random_state=0).fit(X[:50])
print("\nNystroem(500) on 50 samples keeps", len(nys.components_),
      "landmarks; warning:", str(w[0].message).splitlines()[0])
r = RBFSampler(gamma="scale", n_components=10,
               random_state=0).fit(X)
print("RBFSampler gamma='scale': _gamma =", round(r._gamma, 4),
      " 1/(p var) =", round(1 / (X.shape[1] * X.var()), 4))

Xd, yd = load_digits(return_X_y=True)
Xd = Xd[:1347]                    # as many rows as the train set
try:
    GaussianRandomProjection(random_state=0).fit(Xd)
except ValueError as e:
    print("\nn_components='auto' on the digits:", e)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    GaussianRandomProjection(100, random_state=0).fit(Xd)
print("n_components=100 > p = 64 gives", w[0].category.__name__)
