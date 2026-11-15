# Hand checks for the three class A problems of Chapter 41 (and the slips named in the instructor notes)
import numpy as np
from math import comb
from sklearn.metrics import (silhouette_samples, adjusted_rand_score, rand_score,
                             calinski_harabasz_score, davies_bouldin_score)

print("== 41A.1  silhouette of p = 3")
x = np.array([1, 2, 3, 5, 6, 8, 10, 11.]); lab = np.array([0, 0, 0, 1, 1, 1, 2, 2]); p = 3.0
a = np.mean([abs(p - q) for q in (1, 2)])
b = {k: np.mean([abs(p - q) for q in x[lab == k]]) for k in (1, 2)}
bb = min(b.values()); s = (bb - a) / max(a, bb)
print("a =", a, " b(cluster 2) = %.4f  b(cluster 3) = %.4f  b = %.4f  s = %.4f" % (b[1], b[2], bb, s))
print("silhouette_samples:", round(silhouette_samples(x[:, None], lab)[2], 4))
a_wrong = np.mean([abs(p - q) for q in (1, 2, 3)])
print("slip: own cluster averaged over 3 (self included): a = %.4f, s = %.4f" % (a_wrong, (bb - a_wrong) / max(a_wrong, bb)))
bmin = min(abs(p - q) for q in x[lab == 1]); print("slip: nearest neighbour instead of mean distance: b = %.1f, s = %.4f" % (bmin, (bmin - a) / max(a, bmin)))
print("distances of p to cluster 2:", [float(abs(p - q)) for q in x[lab == 1]], " to cluster 3:", [float(abs(p - q)) for q in x[lab == 2]])

print("\n== 41A.2  Rand and adjusted Rand")
N = np.array([[4, 1], [1, 4]]); n = 10
c2 = lambda m: sum(comb(int(v), 2) for v in np.ravel(m))
P11, Pt, Pc = c2(N), c2(N.sum(1)), c2(N.sum(0)); P10, P01 = Pt - P11, Pc - P11; P00 = comb(n, 2) - P11 - P10 - P01
print("cell pairs C(n_ij,2):", [[comb(int(v), 2) for v in r] for r in N], " class pairs", Pt, " cluster pairs", Pc, " C(10,2) =", comb(n, 2))
print("P11 P10 P01 P00 =", P11, P10, P01, P00, " sum", P11 + P10 + P01 + P00)
E = Pt * Pc / comb(n, 2); mx = (Pt + Pc) / 2
print("RI = %d/45 = %.4f   E = %.4f   max = %.1f   ARI = %.4f" % (P11 + P00, (P11 + P00) / 45, E, mx, (P11 - E) / (mx - E)))
print("pair form: 2(P11 P00 - P10 P01) / [...] = %.4f" % (2 * (P11 * P00 - P10 * P01) / ((P11 + P10) * (P10 + P00) + (P11 + P01) * (P01 + P00))))
t = np.repeat([0, 1], 5); u = np.array([0] * 4 + [1] + [0] + [1] * 4)
print("sklearn: RI %.4f  ARI %.4f" % (rand_score(t, u), adjusted_rand_score(t, u)))
print("slip: RI taken as the answer: 0.6444, which looks good although ARI is 0.28; expected RI of independent labellings with these sizes is", round((Pt * Pc / 45 + (45 - Pt - Pc + Pt * Pc / 45)) / 45, 4))

print("\n== 41A.3  CH and DB")
x6 = np.array([1, 2, 3, 7, 8, 9.]); l6 = np.array([0, 0, 0, 1, 1, 1]); m = x6.mean()
mu = [float(x6[l6 == k].mean()) for k in (0, 1)]
SST = ((x6 - m) ** 2).sum(); SSW = sum(((x6[l6 == k] - mu[k]) ** 2).sum() for k in (0, 1)); SSB = sum(3 * (mu[k] - m) ** 2 for k in (0, 1))
print("overall mean", m, " cluster means", mu, " SST", SST, " SSW", SSW, " SSB", SSB, " sum", SSW + SSB)
CH = (SSB / 1) / (SSW / 4); print("CH = (%.0f/1)/(%.0f/4) = %.1f   sklearn %.1f" % (SSB, SSW, CH, calinski_harabasz_score(x6[:, None], l6)))
S = [np.abs(x6[l6 == k] - mu[k]).mean() for k in (0, 1)]; d = abs(mu[0] - mu[1])
R = (S[0] + S[1]) / d; print("S_k =", np.round(S, 4), " centre distance", d, " ratio = %.4f   DB = %.4f   sklearn %.4f" % (R, R, davies_bouldin_score(x6[:, None], l6)))
print("slip: divide SSW by n = 6 instead of n-K = 4: CH = %.1f" % (SSB / (SSW / 6)))
