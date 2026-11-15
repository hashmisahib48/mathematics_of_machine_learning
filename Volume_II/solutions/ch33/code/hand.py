# Hand checks for Problems 33A.1 - 33A.3 and the slips quoted in the notes
import numpy as np
from itertools import combinations

print("== 33A.1  K = 6, class sizes 400 300 200 100 50 50")
sz = [400, 300, 200, 100, 50, 50]; n = sum(sz); K = 6
print("n =", n, " OvR problems", K, " rows each", n, " rows in all", K * n)
pr = list(combinations(range(K), 2))
rows = [sz[j] + sz[k] for j, k in pr]
print("OvO problems:", len(pr), " rows of each:", rows)
print("OvO rows in all:", sum(rows), " (K-1) n =", (K - 1) * n)
print("by class: each class is in K-1 = 5 pairs, so rows =", sum((K - 1) * s for s in sz))
c_ovr = K * n ** 2; c_ovo = sum(r ** 2 for r in rows)
print("squared-rows cost: OvR", c_ovr, " OvO", c_ovo, " ratio", round(c_ovo / c_ovr, 4), " 2(K-1)/K^2 =", round(2 * (K - 1) / K ** 2, 4))
print("squares of the rows:", [r ** 2 for r in rows])
print("equal classes would give OvO/OvR =", round(2 * (K - 1) / K ** 2, 4))

print("\n== 33A.2  f01 = 0.8, f12 = 0.3, f02 = -0.6")
f = {(0, 1): .8, (1, 2): .3, (0, 2): -.6}; v = np.zeros(3); c = np.zeros(3)
for (j, k), s in f.items():
    w = k if s > 0 else j; v[w] += 1; c[j] -= s; c[k] += s
    print(f"pair ({j},{k}): f = {s:+.1f} -> vote for class {w}")
print("votes:", v, " c:", c.round(4))
den = 3 * (np.abs(c) + 1); print("3(|c|+1):", den.round(4), " c/(3(|c|+1)):", (c / den).round(4))
F = v + c / den; print("F:", F.round(4), " class", F.argmax())
print("c sums to", c.sum().round(12), "(every pair adds +f and -f)")

print("\n== 33A.3  code book of Example 33.5-3, s = (-0.9 0.5 0.8 0.7 -0.3 0.6)")
M = np.array([[1, 1, 1, 1, 1, 1], [1, 1, -1, -1, -1, -1], [-1, -1, 1, 1, -1, -1], [-1, -1, -1, -1, 1, 1]])
s = np.array([-0.9, 0.5, 0.8, 0.7, -0.3, 0.6])
for k, name in enumerate("ABCD"):
    t = (s - M[k]) ** 2
    print(name, "terms", t.round(2), " sum", t.sum().round(2), " distance", np.sqrt(t.sum()).round(3))
sg = np.sign(s).astype(int); print("signs of s:", sg)
print("sign disagreements:", [(sg != M[k]).sum() for k in range(4)])
ham = (M[:, None] != M[None]).sum(2); print("Hamming matrix:\n", ham)
print("smallest distance", ham[np.triu_indices(4, 1)].min(), " corrects", (ham[np.triu_indices(4, 1)].min() - 1) // 2)
print("wrong signs of s against C:", np.flatnonzero(sg != M[2]) + 1, " against A:", np.flatnonzero(sg != M[0]) + 1)

print("\n== row sums of the table of Problem 33A.1 (upper triangle, class j against later classes k)")
for j in range(K):
    r = [sz[j] + sz[k] for k in range(j + 1, K)]
    print(j + 1, r, " sum", sum(r), " squares sum", sum(t * t for t in r))
print("total rows", sum(sz[j] + sz[k] for j, k in pr), " total squares", sum((sz[j] + sz[k]) ** 2 for j, k in pr))

print("\n== slip in 33A.2: signs of the confidences reversed")
cr = -c; Fr = v + cr / (3 * (np.abs(cr) + 1)); print("c' =", cr.round(4), " F' =", Fr.round(4), " class", Fr.argmax())
