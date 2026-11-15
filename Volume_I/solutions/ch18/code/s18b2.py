# Problem 18B.2: rank of the pooled covariance and shrinkage.
import numpy as np

P, N = 50, 40                         # features, training samples
rng = np.random.default_rng(0)        # as in Listing 18.4
idx = np.arange(P)
L = np.linalg.cholesky(0.6 ** np.abs(idx[:, None] - idx[None, :]))
delta = 0.3 * (-1.0) ** idx
y = np.arange(N) % 2
X = rng.normal(size=(N, P)) @ L.T + y[:, None] * delta

# pooled within-class covariance, divisor n_k, Eq. 18.3-10
Sw = sum((y == k).mean() * np.cov(X[y == k].T, bias=True)
         for k in (0, 1))
ev = np.sort(np.linalg.eigvalsh(Sw))[::-1]
rank = int((ev > 1e-10 * ev[0]).sum())
tr = np.trace(Sw) / P
print("rank of the pooled covariance:", rank, " n - K =", N - 2)
print(f"largest eigenvalue {ev[0]:.4f}   smallest {ev[-1]:.1e}")
print(f"tr(Sw)/p = {tr:.4f}")

for a in (0.0, 0.1, 0.5):
    Sa = (1 - a) * Sw + a * tr * np.eye(P)       # Eq. 18.5-3
    e = np.linalg.eigvalsh(Sa)
    cond = "singular" if e[0] < 1e-12 else f"{e[-1] / e[0]:.1f}"
    print(f"a = {a}: min {abs(e[0]):.4f} max {e[-1]:.4f} "
          f" a*tr/p {a * tr:.4f}  cond {cond}")
