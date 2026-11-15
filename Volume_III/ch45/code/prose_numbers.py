# Numbers quoted in Chapter 45 that no other program prints.
import json
import numpy as np
from scipy.stats import chi2
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.random_projection import (SparseRandomProjection,
    johnson_lindenstrauss_min_dim as jl)

N = {}
f = lambda e: e**2 / 2 - e**3 / 3
print("-- Eq. 45.3-5, k >= 4 ln n / (eps^2/2 - eps^3/3)")
for n, e in ((1000, 0.1), (1000, 0.5), (10**6, 0.1), (10**6, 0.5)):
    print(f"n = {n:8d} eps {e}: formula {4*np.log(n)/f(e):10.2f}"
          f"  jl {int(jl(n, eps=e))}")
print("growth: eps 0.1 -> 0.05 multiplies k by",
      round(float(jl(1000, eps=0.05) / jl(1000, eps=0.1)), 3),
      "; n 10^3 -> 10^6 by", round(float(jl(1e6) / jl(1e3)), 3))
print("\n-- Chernoff bound of Eq. 45.3-3 against the exact tails")
for k, e in ((30, 0.5), (100, 0.3), (300, 0.2)):
    up = chi2.sf((1 + e) * k, k)
    lo = chi2.cdf((1 - e) * k, k)
    print(f"k = {k:3d} eps {e}: P(q>=1+eps) {up:.2e}  P(q<=1-eps) "
          f"{lo:.2e}  bound {np.exp(-k * f(e) / 2):.2e}")

print("\n-- Eq. 45.3-7: variance of the squared length")
p, k, reps = 100, 10, 4000
flat, spiky = np.ones(p) / 10, np.zeros(p)
spiky[0] = 1.0
for name, x in (("flat", flat), ("spiky", spiky)):
    for s in (3, 10):
        q = np.array([(SparseRandomProjection(
            k, density=1 / s, random_state=r).fit_transform(
            x[None]) ** 2).sum() for r in range(reps)])
        th = (2 + (s - 3) * (x**4).sum() / (x @ x) ** 2) / k
        print(f"{name:5s} s = {s:2d}: variance {q.var():.3f}"
              f" (formula {th:.3f}), mean {q.mean():.3f}")

print("\n-- Eq. 45.4-5: variance of one random feature product")
rng = np.random.RandomState(0)
for d in (0.0, 0.5, 1.0, 2.0):
    w = rng.normal(scale=1.0, size=400000)      # gamma = 0.5
    b = rng.uniform(0, 2 * np.pi, 400000)
    t = 2 * np.cos(b) * np.cos(w * d + b)       # x = 0, z = d
    kk = np.exp(-0.5 * d * d)
    print(f"d = {d}: k {kk:.4f} mean {t.mean():.4f} variance "
          f"{t.var():.4f} formula {1 + kk**4 / 2 - kk**2:.4f}")

print("\n-- Nystroem: exact for rank <= m, error is PSD")
X = rng.normal(size=(40, 3))
K = X @ X.T                         # linear kernel, rank 3
Z = Nystroem(kernel="linear", n_components=3,
             random_state=0).fit_transform(X)
print("rank-3 linear kernel, m = 3: max |K - ZZ^T| =",
      f"{np.abs(K - Z @ Z.T).max():.1e}")
Xd = load_digits(return_X_y=True)[0][:400] / 16
Kd = rbf_kernel(Xd, gamma=0.11)
nys = Nystroem(gamma=0.11, n_components=40, random_state=0).fit(Xd)
Zn = nys.transform(Xd)
E = Kd - Zn @ Zn.T
I = nys.component_indices_
print("digits, m = 40: min eigenvalue of K - ZZ^T",
      f"{np.linalg.eigvalsh(E).min():.1e}; error on landmark block "
      f"{np.abs(E[np.ix_(I, I)]).max():.1e}; max diagonal error "
      f"{np.diag(E).max():.3f}")

print("\n-- relative Frobenius error of the Gram matrix, digits, "
      "n = 1347 train")
Xt = load_digits(return_X_y=True)[0]
from sklearn.model_selection import train_test_split
Xtr = train_test_split(Xt / 16, test_size=0.25, random_state=0,
                       stratify=load_digits(return_X_y=True)[1])[0]
g = 1 / (64 * Xtr.var())
Kt = rbf_kernel(Xtr, gamma=g)
for D in (10, 100, 800):
    r = {}
    for nm, cls in (("RBFSampler", RBFSampler),
                    ("Nystroem", Nystroem)):
        Zt = cls(gamma=g, n_components=D,
                 random_state=0).fit_transform(Xtr)
        r[nm] = np.linalg.norm(Kt - Zt @ Zt.T) / np.linalg.norm(Kt)
    print(f"D = {D:3d}: RBFSampler {r['RBFSampler']:.4f}"
          f" (x sqrt(D) = {r['RBFSampler']*np.sqrt(D):.3f})"
          f"  Nystroem {r['Nystroem']:.4f}")
    N[D] = r
json.dump({str(k): v for k, v in N.items()},
          open("verify/prose_numbers.json", "w"))
