# Example 44.8-1: classical MDS of four villages, by hand in NumPy.
import numpy as np
from sklearn.manifold import ClassicalMDS

np.set_printoptions(precision=4, suppress=True)
# straight-line distances between villages A, B, C, D (3-4-5 sides)
D = np.array([[0, 4, 5, 3],
              [4, 0, 3, 5],
              [5, 3, 0, 4],
              [3, 5, 4, 0]], float)
# the same villages when only the four side roads exist (geodesic)
G = D.copy()
G[0, 2] = G[2, 0] = G[1, 3] = G[3, 1] = 7.0

n = 4
J = np.eye(n) - np.ones((n, n)) / n             # Eq. 44.3-2
for name, M in (("straight-line", D), ("side roads", G)):
    B = -0.5 * J @ (M ** 2) @ J
    w, V = np.linalg.eigh(B)
    w, V = w[::-1], V[:, ::-1]                  # descending order
    print(name, "- eigenvalues of B:", (w.round(4) + 0.0))
    Z = V[:, :2] * np.sqrt(w[:2])               # Eq. 44.3-3
    print("Z =\n", Z)
    ref = ClassicalMDS(2, metric="precomputed").fit(M)
    print("sklearn eigenvalues_:", ref.eigenvalues_.round(4))
    same = np.allclose(np.abs(Z), np.abs(ref.embedding_))
    print("same embedding up to sign:", same)
    print("fitted distances:",
          np.linalg.norm(Z[0] - Z[2]).round(4),
          np.linalg.norm(Z[0] - Z[1]).round(4))
