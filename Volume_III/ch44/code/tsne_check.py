# t-SNE: our P, Q, KL and gradient against scikit-learn.
import numpy as np
from scipy.optimize import approx_fprime
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X = load_digits().data[:200]
n, perp = len(X), 30.0
D2 = squareform(pdist(X, "sqeuclidean"))

P1 = np.zeros((n, n))                           # Eq. 44.7-1
for i in range(n):
    lo, hi = 0.0, 1e3
    for _ in range(100):                        # bisection on beta
        beta = 0.5 * (lo + hi)
        d = D2[i].copy()
        d[i] = np.inf                           # no self-pair
        e = np.exp(-beta * (d - d.min()))
        p = e / e.sum()
        H = -(p[p > 0] * np.log(p[p > 0])).sum()
        lo, hi = (beta, hi) if np.exp(H) > perp else (lo, beta)
    P1[i] = p
P = (P1 + P1.T) / (2 * n)                       # Eq. 44.7-3

def kl_and_grad(z):                             # Eqs. 44.7-4 to 6
    Z = z.reshape(n, 2)
    W = 1 / (1 + squareform(pdist(Z, "sqeuclidean")))
    np.fill_diagonal(W, 0)
    Q = W / W.sum()
    m = ~np.eye(n, dtype=bool)
    kl = (P[m] * np.log(P[m] / Q[m])).sum()
    F = ((P - Q) * W)[:, :, None] * (Z[:, None] - Z[None])
    g = 4 * F.sum(axis=1)
    return kl, g.ravel()

ts = TSNE(2, perplexity=perp, method="exact", init="pca",
          random_state=0).fit(X)
kl, _ = kl_and_grad(ts.embedding_.ravel())
print(f"our KL {kl:.4f}, kl_divergence_ {ts.kl_divergence_:.4f}")
z0 = np.random.RandomState(1).normal(size=2 * n)
g = kl_and_grad(z0)[1]
gn = approx_fprime(z0, lambda z: kl_and_grad(z)[0], 1e-6)
print("gradient vs finite differences: max |diff| =",
      f"{np.abs(g - gn).max():.1e}", " max |g| =",
      f"{np.abs(g).max():.1e}")
print("auto learning rate, n = 1347 and 20000:",
      [max(m / 12 / 4, 50) for m in (1347, 20000)])
