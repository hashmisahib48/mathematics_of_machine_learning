# The hand calculation of Example 39.6-1 in NumPy.
import itertools
import numpy as np
from scipy.linalg import eigh
from data import toy_adjacency

W = toy_adjacency()                  # two triangles, one link (3-4)
d = W.sum(axis=1)                    # degrees
L = np.diag(d) - W                   # Eq. 39.3-2
print("degrees", d)
rows = []
for r in range(1, 6):                # every split (A, B), 1 in A
    for A in itertools.combinations(range(6), r):
        if 0 in A:
            B = [i for i in range(6) if i not in A]
            cut = W[np.ix_(A, B)].sum()
            vA, vB = d[list(A)].sum(), d[B].sum()
            ncut = cut * (1 / vA + 1 / vB)       # Eq. 39.3-4
            rows.append((ncut, cut, A))
for ncut, cut, A in sorted(rows)[:3]:
    print("A =", [a + 1 for a in A], "cut", cut, "Ncut",
          round(ncut, 4))
lam, F = eigh(L, np.diag(d))         # L f = lambda D f (Eq. 39.3-7)
f = F[:, 1] / F[0, 1]
print("lambda_2 =", lam[1].round(4), "f =", f.round(4))
