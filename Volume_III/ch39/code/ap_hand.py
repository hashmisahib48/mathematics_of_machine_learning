# Affinity propagation by hand (Example 39.6-2), without damping.
import numpy as np
from sklearn.cluster import AffinityPropagation

x = np.array([0.0, 1.0, 3.0])
S0 = -(x[:, None] - x[None]) ** 2     # s(i,k), Eq. 39.4-1
S = S0.copy()
np.fill_diagonal(S, -6.0)             # preference p = -6
A = np.zeros((3, 3))
for rnd in (1, 2):
    AS = A + S                        # Eq. 39.4-3
    R = S - np.array([[np.delete(AS[i], k).max() for k in range(3)]
                      for i in range(3)])
    P = np.maximum(R, 0)
    np.fill_diagonal(P, np.diag(R))   # r(k,k) kept as it is
    col = P.sum(axis=0)
    A = np.minimum(0, col - P)        # Eq. 39.4-4
    np.fill_diagonal(A, col - np.diag(R))   # Eq. 39.4-5
    print("round", rnd, "R =", R.tolist())
    print("        A =", A.tolist())
    print("        a_kk + r_kk =", (np.diag(A) + np.diag(R)))
for p in (None, -3.0, -6.0):
    m = AffinityPropagation(affinity="precomputed", preference=p,
                            random_state=0).fit(S0)
    print("preference", p, "exemplars", m.cluster_centers_indices_)
