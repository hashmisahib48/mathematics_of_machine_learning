# Problem 39C.1: equal rows in the null space, and what a weak link does to them
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ch39/code'))
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from data import toy_adjacency

rng = np.random.default_rng(0)
for eps in (0.0, 0.01, 0.1, 1.0):                  # weight of the link (3, 4)
    W = toy_adjacency(); W[2, 3] = W[3, 2] = eps
    d = W.sum(1); L = np.diag(d) - W
    lam, F = eigh(L, np.diag(d))                   # L f = lambda D f
    U = F[:, :2]                                   # rows u_i of the embedding, K = 2
    if eps == 0:                                   # any basis of the null space will do:
        Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))   # rotate it at random
        U = U @ Q
    within = max(np.linalg.norm(U[i] - U[j]) for A in (range(3), range(3, 6)) for i in A for j in A)
    between = np.linalg.norm(U[:3].mean(0) - U[3:].mean(0))
    km = KMeans(2, n_init=1, init='k-means++', random_state=0).fit(U)
    print(f"eps = {eps:5.2f}  lambda_2 = {abs(lam[1]):.4f}  within-triangle spread {within:.4f}"
          f"  distance between triangles {between:.4f}  ratio {within / between:.4f}"
          f"  | k-means: labels {km.labels_.tolist()} inertia {km.inertia_:.2e} iterations {km.n_iter_}")
    if eps == 0: print("   rows of U (random basis of the null space):", U.round(4).tolist())
# f'Lf = 0 for a vector that is constant on each piece, and > 0 once it differs across an edge
W = toy_adjacency(); W[2, 3] = W[3, 2] = 0; L = np.diag(W.sum(1)) - W
for f in ([2, 2, 2, -5, -5, -5], [1, 1, 1.01, 3, 3, 3]):
    f = np.array(f, float); print("f =", f.tolist(), " f'Lf =", round(float(f @ L @ f), 6))
