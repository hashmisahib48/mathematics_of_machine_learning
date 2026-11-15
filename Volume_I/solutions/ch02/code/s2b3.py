# Problem 2B.3: the best rank-k copy of one digit image
import numpy as np
from sklearn.datasets import load_digits

A = load_digits().images[0]                       # an 8 x 8 matrix
U, s, Vt = np.linalg.svd(A)
print("shape:", A.shape, " rank:", np.linalg.matrix_rank(A))
print("singular values:", np.round(s, 3))
nA = np.linalg.norm(A)                            # Frobenius norm
tot = np.sqrt((s ** 2).sum())
print("||A||_F =", round(nA, 4), " sqrt(sum s^2) =", round(tot, 4))
print(" k  rel. error   Eq. 2.10-5 (from sigma)")
for k in range(1, 5):
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]              # Eq. 2.10-4
    direct = np.linalg.norm(A - Ak) / nA
    formula = np.sqrt((s[k:] ** 2).sum()) / np.sqrt((s ** 2).sum())
    same = np.isclose(direct, formula)
    print(f"{k:2d} {direct:10.4f} {formula:14.4f}   {same}")
kept = np.cumsum(s ** 2)[:4] / (s ** 2).sum()
print("energy kept by k = 1..4:", np.round(kept, 4))
