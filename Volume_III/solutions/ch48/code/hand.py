# Hand checks for the Chapter 48 solutions (Problems 48A.1 - 48A.3).
import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import LocalOutlierFactor

print("== 48A.1  isolation scores for psi = 16")
def c(m):                                   # Eq. 48.4-1 as the library codes it
    return 0 if m <= 1 else 1 if m == 2 else 2 * (np.log(m - 1) + np.euler_gamma) - 2 * (m - 1) / m
print("ln 15 =", round(float(np.log(15)), 4), " + Euler gamma", round(float(np.euler_gamma), 4), "=", round(float(np.log(15) + np.euler_gamma), 4))
print("2 x that =", round(float(2 * (np.log(15) + np.euler_gamma)), 4), " 2*15/16 =", 2 * 15 / 16)
print("c(16) =", round(float(c(16)), 4))
H15 = sum(1 / i for i in range(1, 16)); H16 = H15 + 1 / 16
print("exact 2(H_16 - 1) =", round(2 * (H16 - 1), 4), "  (H_16 =", round(H16, 4), ")")
for h in (2, 4, 6):
    e = h / c(16)
    print(f"E h = {h}: E h / c = {e:.4f}   s = 2^(-{e:.4f}) = {2 ** -e:.4f}   score_samples = {-2 ** -e:.4f}   flagged: {2 ** -e > 0.5}")
print("E h at s = 0.5 is c(16) =", round(float(c(16)), 4))

print("\n== 48A.2  LOF of 0, 1, 2, 4, 10 with k = 2")
x = np.array([0, 1, 2, 4, 10.]); n = 5
D = np.abs(x[:, None] - x[None]); Dn = D + np.diag([np.inf] * n)
print("pairwise distances:\n", D.astype(int))
lib = LocalOutlierFactor(n_neighbors=2).fit(x[:, None])
print("library LOF:", (-lib.negative_outlier_factor_).round(4).tolist())
def lof(nbr):
    dk = np.array([max(D[a, b] for b in nbr[a]) for a in range(n)])
    lrd = np.array([1 / np.mean([max(dk[b], D[a, b]) for b in nbr[a]]) for a in range(n)])
    lo = np.array([np.mean([lrd[b] for b in nbr[a]]) / lrd[a] for a in range(n)])
    return dk, lrd, lo
# two nearest neighbours; point 2 (index 2) has a tie between 0 and 4 (both at distance 2)
base = {0: [1, 2], 1: [0, 2], 3: [2, 1], 4: [3, 2]}
for name, nb2 in (("neighbours of 2 = {1, 0}", [1, 0]), ("neighbours of 2 = {1, 4}", [1, 3])):
    nbr = dict(base); nbr[2] = nb2
    dk, lrd, lo = lof(nbr)
    print(name, " d2 =", dk.tolist(), " lrd =", lrd.round(4).tolist(), " LOF =", lo.round(4).tolist())
# the original definition: all points within the k-distance
dk = np.array([np.sort(Dn[a])[1] for a in range(n)])
nbr = {a: [b for b in range(n) if b != a and D[a, b] <= dk[a]] for a in range(n)}
print("tie-inclusive N_k:", {int(x[a]): [int(x[b]) for b in nbr[a]] for a in range(n)})
dk2, lrd, lo = lof(nbr)
print("tie-inclusive: d2 =", dk2.tolist(), " lrd =", lrd.round(4).tolist(), " LOF =", lo.round(4).tolist())
from sklearn.neighbors import NearestNeighbors
# the neighbours the library picks (self removed; the tie of the point 2 goes to 0)
ind = NearestNeighbors(n_neighbors=3).fit(x[:, None]).kneighbors(x[:, None], return_distance=False)[:, 1:]
print("library neighbour indices (self removed):", {int(x[a]): [int(x[b]) for b in ind[a]] for a in range(n)})
print("\nthe numbers of the solution (neighbours of 2 are 1 and 0):")
nbr = dict(base); nbr[2] = [1, 0]; dk, lrd, lo = lof(nbr)
for a in range(n):
    rr = [(int(x[b]), float(max(dk[b], D[a, b]))) for b in nbr[a]]
    print(f"x={x[a]:4.0f} N={[int(x[b]) for b in nbr[a]]} d2={dk[a]:.0f} reach={rr} mean={np.mean([r[1] for r in rr]):.4f} lrd={lrd[a]:.4f}")
print("LOF(10) = mean(lrd(4), lrd(2)) / lrd(10) =", round(float((lrd[3] + lrd[2]) / 2 / lrd[4]), 4), "= 56/15 =", round(56 / 15, 4))
print("LOF(4)  = mean(lrd(2), lrd(1)) / lrd(4)  =", round(float((lrd[2] + lrd[1]) / 2 / lrd[3]), 4))
print("LOF(0)  = mean(lrd(1), lrd(2)) / lrd(0)  =", round(float((lrd[1] + lrd[2]) / 2 / lrd[0]), 4))
print("LOF(1)  = mean(lrd(0), lrd(2)) / lrd(1)  =", round(float((lrd[0] + lrd[2]) / 2 / lrd[1]), 4))
print("LOF(2)  = mean(lrd(1), lrd(0)) / lrd(2)  =", round(float((lrd[1] + lrd[0]) / 2 / lrd[2]), 4))
print("exact fractions: lrd(4) =", [round(float(v), 4) for v in (lrd[3],)], " 1/lrd(4)=", round(float(1 / lrd[3]), 4), " 1/lrd(10) =", round(float(1 / lrd[4]), 4))

print("\n== 48A.3  Mahalanobis distance")
mu = np.array([70, 2.5]); S = np.array([[16, 1.2], [1.2, 0.25]]); q = np.array([78, 3.5]); d = q - mu
print("deviation", d.tolist(), " det S =", round(float(np.linalg.det(S)), 4), " 16*0.25 - 1.2^2 =", 16 * 0.25 - 1.2 ** 2)
Si = np.linalg.inv(S); print("S^-1 =", Si.round(4).tolist(), " exact: [[0.25, -1.2], [-1.2, 16]] / 2.56 =", (np.array([[0.25, -1.2], [-1.2, 16]]) / 2.56).round(4).tolist())
print("terms: 64 *", round(float(Si[0, 0]), 4), "=", round(float(64 * Si[0, 0]), 4), "; 2*8*1*", round(float(Si[0, 1]), 4), "=", round(float(2 * 8 * Si[0, 1]), 4), "; 1 *", round(float(Si[1, 1]), 4), "=", round(float(Si[1, 1]), 4))
print("d^2 =", round(float(d @ Si @ d), 4))
rho = 1.2 / (4 * 0.5); z1, z2 = 8 / 4, 1 / 0.5
print("sd: 4, 0.5; rho =", rho, " z =", z1, z2, " d^2 = (z1^2 - 2 rho z1 z2 + z2^2)/(1-rho^2) =", round((z1 ** 2 - 2 * rho * z1 * z2 + z2 ** 2) / (1 - rho ** 2), 4), " numerator", z1 ** 2 - 2 * rho * z1 * z2 + z2 ** 2, " 1-rho^2 =", 1 - rho ** 2)
print("chi2(2) 97.5% point:", round(float(chi2.ppf(0.975, 2)), 4), " = -2 ln 0.025 =", round(float(-2 * np.log(0.025)), 4))
print("Euclidean squared:", float(d @ d), "; if the vibration is ignored d^2_temp =", z1 ** 2)
print("p-value of d^2 = 5:", round(float(chi2.sf(5.0, 2)), 4), " = exp(-5/2) =", round(float(np.exp(-2.5)), 4))
# a reading with the same Euclidean distance but different d^2
for dev in ([8.06, 0.0], [0.0, 8.06]):
    dv = np.array(dev); print("deviation", dev, "euclid^2", round(float(dv @ dv), 1), "d^2 =", round(float(dv @ Si @ dv), 2))
print("ignoring the off-diagonal term: d^2 =", (8 / 4) ** 2 + (1 / 0.5) ** 2, "  (z1^2 + z2^2 with sd 4 and 0.5)  > 7.3778 would flag the reading wrongly")
