# Eight customers: the scores of the text, step by step.
import numpy as np
from math import comb
from scipy.spatial.distance import cdist

X = np.array([[1, 1], [2, 3], [3, 1], [4, 3],
              [7, 2], [8, 4], [3, 7], [5, 8]], float)
seg = np.array([0, 0, 0, 1, 1, 1, 2, 2])   # shop's segments (truth)
clu = np.array([0, 0, 0, 0, 1, 1, 2, 2])   # k-means, K = 3
n = len(seg)
N = np.zeros((3, 3), int)
np.add.at(N, (seg, clu), 1)                 # contingency table n_ij
ni, nj = N.sum(axis=1), N.sum(axis=0)       # n_i., n_.j
pair = lambda m: sum(comb(int(v), 2) for v in np.ravel(m))
P11, Pt, Pc = pair(N), pair(ni), pair(nj)   # Eq. 41.4-1
P10, P01 = Pt - P11, Pc - P11
P00 = comb(n, 2) - P11 - P10 - P01
E = Pt * Pc / comb(n, 2)                    # Eq. 41.4-3
ari = (P11 - E) / ((Pt + Pc) / 2 - E)       # Eq. 41.4-4
print("pairs P11, P10, P01, P00:", P11, P10, P01, P00)
print(f"Rand {(P11 + P00) / comb(n, 2):.4f}  ARI {ari:.4f}  "
      f"FM {P11 / np.sqrt(Pt * Pc):.4f}")

H = lambda p: -np.sum(p * np.log(p))        # Eq. 41.4-5, in nats
Hc, Hu = H(ni / n), H(nj / n)
i, j = np.nonzero(N)                        # Eq. 41.4-6
MI = np.sum(N[i, j] / n * np.log(n * N[i, j] / (ni[i] * nj[j])))
hom, com = MI / Hc, MI / Hu                 # Eq. 41.4-7
print(f"H(c) {Hc:.4f}  H(u) {Hu:.4f}  MI {MI:.4f}  "
      f"NMI {MI / ((Hc + Hu) / 2):.4f}")
print(f"homogeneity {hom:.4f}  completeness {com:.4f}  "
      f"V {2 * hom * com / (hom + com):.4f}")

D = cdist(X, X)                             # Eqs. 41.3-1, 41.3-2
s = np.zeros(n)
for p in range(n):
    own = (clu == clu[p]) & (np.arange(n) != p)
    ap = D[p, own].mean()
    bp = min(D[p, clu == k].mean() for k in set(clu) if k != clu[p])
    s[p] = (bp - ap) / max(ap, bp)
print("silhouette:", s.round(3), " mean", s.mean().round(4))
