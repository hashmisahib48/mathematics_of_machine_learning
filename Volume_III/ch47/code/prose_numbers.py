# Numbers quoted in the text of Chapter 47 (not shown as a listing).
import time
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def waits(n, seed):
    rng = np.random.default_rng(seed)
    short = rng.random(n) < 0.35
    t = np.where(short, rng.normal(54, 5, n), rng.normal(80, 6, n))
    return t[:, None]


def truth(x):
    return 0.35 * norm.pdf(x, 54, 5) + 0.65 * norm.pdf(x, 80, 6)


X, X_test = waits(200, 1), waits(5000, 2)
n, sd = len(X), X.std()
print("sample: mean %.2f sd %.2f min %.1f max %.1f" %
      (X.mean(), sd, X.min(), X.max()))

print("\n-- kernels in one dimension, h = 1 (numerical integrals)")
xs = np.linspace(-30, 30, 600001)
for k in ["gaussian", "tophat", "epanechnikov", "exponential",
          "linear", "cosine"]:
    v = np.exp(KernelDensity(kernel=k).fit([[0.0]])
               .score_samples(xs[:, None]))
    print(f"{k:13s} area {np.trapezoid(v, xs):.4f}  K(0) {v[300000]:.4f}"
          f"  sd {np.sqrt(np.trapezoid(v * xs**2, xs)):.4f}"
          f"  R(K) {np.trapezoid(v**2, xs):.4f}")

print("\n-- same smoothing, different kernels: h times sd of the kernel")
print("gaussian 2.5 x 1.0 = %.3f ; epanechnikov 5.75 x 0.4472 = %.3f" % (2.5, 5.75 * 0.4472))

print("\n-- AMISE bandwidth for the true mixture")
g = np.linspace(0, 140, 140001)
f2 = np.gradient(np.gradient(truth(g), g), g)
Rf2 = np.trapezoid(f2**2, g)
RK = 1 / (2 * np.sqrt(np.pi))
h_amise = (RK / (Rf2 * n)) ** 0.2
print(f"R(f'') = {Rf2:.3e}   h_AMISE (n = 200) = {h_amise:.3f}")
s_true = np.sqrt(0.35 * (54**2 + 25) + 0.65 * (80**2 + 36)
                 - (0.35 * 54 + 0.65 * 80) ** 2)
print(f"true sd {s_true:.3f};  1.0592 s n^(-1/5) = "
      f"{(4 / 3) ** 0.2 * s_true * n ** -0.2:.3f}")
print(f"Scott factor {n ** -0.2:.4f}  Silverman factor "
      f"{(n * 3 / 4) ** -0.2:.4f}  (p = 1, n = 200)")
raw = KernelDensity(bandwidth=n ** -0.2).fit(X)
print(f"raw Scott factor as h: test mean ln f = "
      f"{raw.score(X_test) / len(X_test):.3f}")

print("\n-- rate exponent 4/(p+4) and sample size for equal error")
for p in [1, 2, 5, 10, 20]:
    print(f"p = {p:2d}  exponent {4 / (p + 4):.3f}  "
          f"n for the error of n = 100 at p = 1: "
          f"{100 ** ((p + 4) / 5):.3g}")

print("\n-- leave-one-out from the full sum (Gaussian, h = 2.5)")
h = 2.5
full = KernelDensity(bandwidth=h).fit(X).score_samples(X)
c0 = 1 / (h * np.sqrt(2 * np.pi))
loo = np.log((n * np.exp(full) - c0) / (n - 1))
for i in [0, 1, 2]:
    k = KernelDensity(bandwidth=h).fit(np.delete(X, i, axis=0))
    print(i, round(float(loo[i]), 6), round(float(k.score_samples(X[i:i + 1])[0]), 6))
print("sum of LOO ln f %.3f; training sum %.3f" % (loo.sum(), full.sum()))
for h in [0.5, 0.05, 0.005]:
    f = KernelDensity(bandwidth=h).fit(X).score(X)
    print(f"h = {h:5.3f}  training mean ln f = {f / n:.4f}")

print("\n-- compact kernels leave gaps")
k = KernelDensity(kernel="epanechnikov", bandwidth=5.75).fit(X)
s = k.score_samples(X_test)
bad = np.isinf(s)
print("test points with density 0:", int(bad.sum()), "of", len(s),
      " range of those points:", X_test[bad].min().round(1),
      X_test[bad].max().round(1))

print("\n-- tolerance, 40 000 samples, 2 000 queries")
Xb, Q = waits(40000, 5), np.linspace(35, 100, 2000)[:, None]
exact = KernelDensity(bandwidth=2.0).fit(Xb)
t0 = time.perf_counter()
fe = np.exp(exact.score_samples(Q))
t_ex = time.perf_counter() - t0
for atol, rtol in [(0, 0.01), (1e-4, 0.0), (0, 0.1)]:
    ap = KernelDensity(bandwidth=2.0, atol=atol, rtol=rtol).fit(Xb)
    t0 = time.perf_counter()
    fa = np.exp(ap.score_samples(Q))
    t_ap = time.perf_counter() - t0
    err = np.abs(fa - fe)
    print(f"atol {atol} rtol {rtol}: max abs err {err.max():.2e}  "
          f"max rel err {(err / fe).max():.2e}  "
          f"bound holds {bool((err <= atol + rtol * fa).all())}  "
          f"time {t_ap:.2f}s vs {t_ex:.2f}s")

print("\n-- sampling inflates the variance")
sm = KernelDensity(bandwidth=2.5).fit(X).sample(200000, random_state=0)
rng = np.random.default_rng(0)                  # Eq. 47.6-1 by hand
own = X[rng.integers(n, size=200000)] + 2.5 * rng.normal(size=(200000, 1))
print(f"own sampler (3 lines) variance {own.var():.2f}")
print(f"sample var {X.var():.2f} + h^2 {2.5**2:.2f} = "
      f"{X.var() + 2.5**2:.2f};  drawn {sm.var():.2f}")

print("\n-- Gaussian KDE is a mixture with n components")
q = 58.0
lhs = KernelDensity(bandwidth=3.0).fit(X).score_samples([[q]])[0]
rhs = np.log(np.mean(norm.pdf(q, X[:, 0], 3.0)))
print(f"{lhs:.6f} {rhs:.6f}")
