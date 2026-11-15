# Example 44.8-3: t-SNE affinities, KL divergence and gradient.
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities

np.set_printoptions(precision=4, suppress=True)
x = np.array([0.0, 1.0, 3.0, 4.0])              # four points
D2 = (x[:, None] - x[None, :]) ** 2             # squared distances
n, perp = len(x), 2.0

def cond_row(i, beta):                          # Eq. 44.7-1
    e = np.exp(-beta * D2[i]); e[i] = 0.0
    return e / e.sum()

def perplexity(p):                              # Eq. 44.7-2
    q = p[p > 0]
    return np.exp(-(q * np.log(q)).sum())

P1 = np.zeros((n, n))
for i in range(n):
    lo, hi = 0.0, 50.0                          # bisection on beta
    for _ in range(60):
        beta = 0.5 * (lo + hi)
        if perplexity(cond_row(i, beta)) > perp:
            lo = beta                           # too flat: sharpen
        else:
            hi = beta
    P1[i] = cond_row(i, beta)
    print(f"point {i+1}: beta = {beta:.4f}, sigma = "
          f"{np.sqrt(1/(2*beta)):.4f}, perplexity = "
          f"{perplexity(P1[i]):.4f}")
P = (P1 + P1.T) / (2 * n)                       # Eq. 44.7-3
print("joint P (x 100):\n", 100 * P)
ref = squareform(_joint_probabilities(D2, perp, 0))
print("matches scikit-learn:", np.allclose(P, ref, atol=1e-4))
Z = np.array([[0, 0], [1, 0], [0, 1], [1, 1.5]])    # a layout
d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1)
W = 1 / (1 + d2); np.fill_diagonal(W, 0)
Q = W / W.sum()                                 # Eq. 44.7-4
m = ~np.eye(n, dtype=bool)
print("KL(P||Q) =", round(float((P[m] * np.log(P[m] / Q[m])).sum()), 4))
F = ((P - Q) * W)[:, :, None] * (Z[:, None] - Z[None])
print("gradient (Eq. 44.7-6):\n", 4 * F.sum(axis=1))
