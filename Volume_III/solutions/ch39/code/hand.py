# Hand checks for the Chapter 39 solutions (problems 39A.1 to 39A.4, 39C.1). Run: python3 hand.py
import sys, os, itertools, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch39/code')
import numpy as np
from scipy.linalg import eigh
from data import toy_adjacency

def lap(W):
    d = W.sum(1); return d, np.diag(d) - W

# ---- 39A.1  path 1-2-3-4
W = np.zeros((4, 4))
for i in range(3): W[i, i + 1] = W[i + 1, i] = 1
d, L = lap(W)
print("39A.1 degrees", d.tolist())
for A in ([0], [0, 1], [0, 1, 2]):
    B = [i for i in range(4) if i not in A]
    cut = W[np.ix_(A, B)].sum(); vA, vB = d[A].sum(), d[B].sum()
    print("  A = %-9s cut %.0f  vol %.0f, %.0f  Ncut = %.0f(1/%.0f+1/%.0f) = %.4f   RatioCut = %.0f(1/%d+1/%d) = %.4f"
          % (str([a + 1 for a in A]), cut, vA, vB, cut, vA, vB, cut * (1 / vA + 1 / vB), cut, len(A), len(B), cut * (1 / len(A) + 1 / len(B))))

# ---- 39A.2  six points, f = (1,1,1,-1,-1,-1)
W6 = toy_adjacency(); d6, L6 = lap(W6)
f = np.array([1, 1, 1, -1, -1, -1.])
print("\n39A.2 degrees", d6.tolist())
edges = [(i + 1, j + 1) for i in range(6) for j in range(i + 1, 6) if W6[i, j]]
print("  edges", edges)
print("  f'Lf = sum over edges w (fi-fj)^2 =", sum((f[i - 1] - f[j - 1]) ** 2 for i, j in edges), " matrix product", f @ L6 @ f)
print("  f'Df =", f @ (d6 * f), "  f'D1 =", f @ d6, "  quotient", round(f @ L6 @ f / (f @ (d6 * f)), 4))
cut = 1; vA = d6[:3].sum(); vB = d6[3:].sum()
print("  vol A, B", vA, vB, " Ncut = 1*(1/7+1/7) =", round(cut * (1 / vA + 1 / vB), 4))
print("  lambda2 = (11 - sqrt 73)/12 =", round((11 - np.sqrt(73)) / 12, 4), " from eigh:", round(eigh(L6, np.diag(d6), eigvals_only=True)[1], 4))
print("  (sqrt(vB/vA)+sqrt(vA/vB))^2 =", (np.sqrt(vB / vA) + np.sqrt(vA / vB)) ** 2)
print("  rows of L f:", (L6 @ f).tolist())

# ---- 39A.3  delete the link (3,4)
Wb = W6.copy(); Wb[2, 3] = Wb[3, 2] = 0
db, Lb = lap(Wb)
print("\n39A.3 eigenvalues of L:", np.round(np.linalg.eigvalsh(Lb), 6) + 0.0)
Ls = np.eye(6) - Wb / np.sqrt(np.outer(db, db))
print("  eigenvalues of L_sym:", np.round(np.linalg.eigvalsh(Ls), 6) + 0.0)
print("  degrees", db.tolist())
# the embedding of Section 39.3 in the basis (1, 1, 1, 1, 1, 1)/sqrt6 and (1,1,1,-1,-1,-1)/sqrt6 of the null space of L_sym
g1 = np.sqrt(db) / np.linalg.norm(np.sqrt(db)); g2 = np.array([1, 1, 1, -1, -1, -1]) * np.sqrt(db) / np.linalg.norm(np.sqrt(db))
print("  null-space check |Ls g|:", np.abs(Ls @ g1).max().round(12), np.abs(Ls @ g2).max().round(12))
U = np.c_[g1 / np.sqrt(db), g2 / np.sqrt(db)]
print("  embedding rows U = D^-1/2 [g1 g2]:", U.round(4).tolist())
print("  indicator basis of L (1_A/sqrt3, 1_B/sqrt3): value", round(1 / np.sqrt(3), 4))
# with the link back in: the rows for K = 2
lam, F = eigh(L6, np.diag(d6))
print("  with the link: lambda", lam[:3].round(4), " f2 =", (F[:, 1] / F[0, 1]).round(4).tolist())

# ---- 39A.4  net similarity, x = 0, 1, 3
x = np.array([0., 1., 3.])
S = -(x[:, None] - x[None]) ** 2 + 0.0
print("\n39A.4 similarities s(i,k) =\n", S.tolist())
def net(E, p):
    tot = p * len(E)
    for i in range(3):
        if i not in E: tot += max(S[i, k] for k in E)
    return tot
sets = [E for r in (1, 2, 3) for E in itertools.combinations(range(3), r)]
for p in (-6.0, -0.5):
    print("  p =", p)
    for E in sets:
        parts = ["s(%d,%d) = %g" % (i + 1, max(E, key=lambda k: S[i, k]) + 1, max(S[i, k] for k in E)) for i in range(3) if i not in E]
        print("    exemplars %-9s net = %s%s%s%g" % (str([e + 1 for e in E]), "+ ".join(parts) if parts else "", " + " if parts else "", "%d(%g) = " % (len(E), p), net(E, p)))
for p in (-6.0, -4.0, -3.0, -1.0, -0.5):
    best = max(sets, key=lambda E: net(E, p)); print("  p = %5.1f  best sets:" % p, [[b + 1 for b in E] for E in sets if abs(net(E, p) - net(best, p)) < 1e-12], " net", net(best, p))
print("  crossings: -5+p = -1+2p at p =", -4, "; -1+2p = 3p at p =", -1)
