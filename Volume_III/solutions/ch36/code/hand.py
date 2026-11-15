# Hand checks for Problems 36A.1 - 36A.3 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 36A.1  x = 0, 2, 6, 12; centres 1 and 10")
x = np.array([0, 2, 6, 12.]); mu = np.array([1., 10.])
d2 = (x[:, None] - mu[None]) ** 2
lab = d2.argmin(1)
print("d2 to centre 1 :", d2[:, 0]); print("d2 to centre 10:", d2[:, 1])
print("labels:", lab + 1, " smaller d2:", d2.min(1), " J =", d2.min(1).sum())
mu1 = np.array([x[lab == k].mean() for k in (0, 1)])
print("new centres:", mu1)
print("J of the same assignment, new centres:", ((x - mu1[lab]) ** 2).sum())
print("Eq. 36.3-2 for cluster 2: cost at 10 =", ((x[2:] - 10) ** 2).sum(),
      " cost at mean 9 =", ((x[2:] - 9) ** 2).sum(), " n_C d^2 =", 2 * (10 - 9) ** 2)
d2b = (x[:, None] - mu1[None]) ** 2
print("round 2: d2 to 1:", d2b[:, 0], " d2 to 9:", d2b[:, 1], " labels:", d2b.argmin(1) + 1)
print("J after reassignment:", d2b.min(1).sum())

print("\n== 36A.2  x = 1, 3, 8, 10; clusters {1,3} and {8,10}")
x = np.array([1, 3, 8, 10.]); g = [x[:2], x[2:]]
m = x.mean(); print("mean of all:", m, " deviations:", x - m)
sst = ((x - m) ** 2).sum(); print("SST =", sst, " terms:", (x - m) ** 2)
cm = [float(a.mean()) for a in g]; print("cluster means:", cm)
ssw = sum(((a - a.mean()) ** 2).sum() for a in g); print("SSW =", ssw)
ssb = sum(len(a) * (a.mean() - m) ** 2 for a in g); print("SSB =", ssb, " terms:", [len(a) * (a.mean() - m) ** 2 for a in g])
print("SSW + SSB =", ssw + ssb)
for a in g:
    pair = ((a[:, None] - a[None]) ** 2).sum()
    print("Eq. 36.3-4: pair sum", pair, " /(2 n_k) =", pair / (2 * len(a)))
print("K=1: J =", sst, " K=n: J = 0;  best split by brute force:")
for s in range(1, 4):
    a, b = x[:s], x[s:]
    print("  split after", s, "points: J =", ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())

print("\n== 36A.3  x = 0, 1, 2, 10; first centre 0")
x = np.array([0, 1, 2, 10.]); D2 = (x - 0) ** 2; p = D2 / D2.sum()
print("D2:", D2, " sum:", D2.sum()); print("p:", p, " fractions: 0, 1/105, 4/105, 100/105")
print("sum p =", p.sum(), " uniform among the other three:", 1 / 3)
print("odds for the far point:", p[3] / (1 / 3), " near points 1 and 2 together:", p[1] + p[2])
for c2 in (1, 2, 10):
    cen = np.array([0., c2]); J = ((x[:, None] - cen[None]) ** 2).min(1)
    print("  second centre", c2, ": D2 of the points", J, " seed inertia", J.sum())

print("\n== 36A.1 is the fixed point the global minimum?  all splits of the sorted points into two groups")
x = np.array([0, 2, 6, 12.])
for s in range(1, 4):
    a, b = x[:s], x[s:]
    print("  groups", a.tolist(), b.tolist(), ": J =", round(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum(), 4))
