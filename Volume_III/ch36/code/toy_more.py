# Further checks on the eight customers: Examples 36.7-2 and 36.7-3.
import itertools
from fractions import Fraction as F
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus

X = np.array([[3, 7], [1, 3], [1, 4], [3, 8],
              [4, 0], [2, 1], [5, 8], [1, 8]], float)
n = len(X)

def inertia(lab):
    return sum(((X[lab == k] - X[lab == k].mean(0)) ** 2).sum()
               for k in set(lab))

# all 127 two-way partitions: the global optimum
best = min((inertia(np.array([(m >> i) & 1 for i in range(n)])), m)
           for m in range(1, 2 ** (n - 1)))
print("exhaustive search, 127 partitions: minimum inertia", best[0])

# Lloyd from every pair of customers used as starting depots
res = {}
for i, j in itertools.combinations(range(n), 2):
    km = KMeans(2, init=X[[i, j]], n_init=1).fit(X)
    res.setdefault(round(km.inertia_, 4), []).append((i + 1, j + 1))
for J, pairs in sorted(res.items()):
    print(f"local minimum {J}: reached from {len(pairs)} of 28 pairs",
          pairs if len(pairs) < 5 else "")

# sum-of-squares split at the optimum and at the poor minimum
def split(lab):
    xb = X.mean(0)
    sst = ((X - xb) ** 2).sum()
    ssw = inertia(lab)
    return sst, ssw, sst - ssw
good = KMeans(2, init=X[[0, 6]], n_init=1).fit(X).labels_
poor = KMeans(2, init=X[[3, 7]], n_init=1).fit(X)
for name, lab in (("optimum", good), ("poor", poor.labels_)):
    sst, ssw, ssb = split(lab)
    print(f"{name}: SST {sst}  SSW {ssw}  SSB {ssb}  sizes",
          np.bincount(lab).tolist())
print("poor centres", poor.cluster_centers_.tolist())

# k-means++: first depot at customer 5 (4, 0); exact D^2 weights
d2 = [F(int(((X[i] - X[4]) ** 2).sum())) for i in range(n)]
tot = sum(d2)
print("D2 to customer 5:", [int(v) for v in d2], "sum", int(tot))
print("probabilities:", [f"{int(v)}/{int(tot)}" for v in d2])
print("P(second depot in the top group {1,4,7,8}) =",
      sum(d2[i] for i in (0, 3, 6, 7)) / tot,
      "=", round(float(sum(d2[i] for i in (0, 3, 6, 7)) / tot), 4))
# scikit-learn's seeding (greedy, 2 + ln 2 -> 2 trials), first = 5
hits = tot_ = 0
for s in range(6000):
    _, idx = kmeans_plusplus(X, 2, random_state=s)
    if idx[0] == 4:
        tot_ += 1
        hits += idx[1] in (0, 3, 6, 7)
print(f"scikit-learn seeding: {hits}/{tot_} = {hits / tot_:.4f}")

# mini-batch update = running mean (one centre, two batches)
mb = MiniBatchKMeans(1, init=np.array([[9.0, 9.0]]), n_init=1,
                     random_state=0, reassignment_ratio=0)
mb.partial_fit(X[:3])
print("after batch 1:", mb.cluster_centers_.round(4).tolist(),
      "mean of batch 1:", X[:3].mean(0).round(4).tolist())
mb.partial_fit(X[3:])
print("after batch 2:", mb.cluster_centers_.round(4).tolist(),
      "mean of all  :", X.mean(0).round(4).tolist())
