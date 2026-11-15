# Numbers quoted in the model answers to the questions for discussion (Q1 - Q6)
import sys, numpy as np, warnings
sys.path.insert(0, '../../ch36/code')
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_wine, load_digits, make_blobs, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
np.set_printoptions(precision=4, suppress=True)
X8 = np.array([[3, 7], [1, 3], [1, 4], [3, 8], [4, 0], [2, 1], [5, 8], [1, 8]], float)

print("== Q1  best centre for the plain distance")
S = X8[[1, 2, 4, 5]]                                   # the southern cluster {2,3,5,6}
f = lambda m: np.linalg.norm(S - m, axis=1).sum()
mean = S.mean(0); med = np.median(S, 0)
m = mean.copy(); its = 0
while True:                                            # Weiszfeld iteration
    w = 1 / np.linalg.norm(S - m, axis=1); new = (w[:, None] * S).sum(0) / w.sum(); its += 1
    if np.linalg.norm(new - m) < 1e-6: break
    m = new
print("cluster", S.tolist(), " mean", mean, " coordinatewise median", med)
print("geometric median", m, "after", its, "Weiszfeld steps; sum of distances: mean", round(f(mean), 4),
      " geometric median", round(f(m), 4), " coordinatewise median", round(f(med), 4))
S2 = np.vstack([S, [[30, 30]]])                         # one outlier added
g = lambda S: (S.mean(0), np.median(S, 0))
print("with outlier (30,30): mean", g(S2)[0], " coordinatewise median", g(S2)[1])
x1 = np.array([0, 2, 6, 12.])
print("1-D, x = 0,2,6,12: sum|x-m| at m=2,4,6:", [float(np.abs(x1 - v).sum()) for v in (2, 4, 6)], " at mean 5:", np.abs(x1 - 5).sum())

print("\n== Q2  centres restricted to samples")
rng = np.random.default_rng(0); ok = True
for _ in range(2000):
    C = rng.normal(size=(rng.integers(3, 12), 2)) * rng.uniform(0.5, 4)
    best = ((C[:, None] - C[None]) ** 2).sum(2).sum(0).argmin()      # best member by exhaustive sum
    near = np.linalg.norm(C - C.mean(0), axis=1).argmin()             # member nearest the mean
    ok &= (best == near)
print("2000 random clusters: best member by sum of squares is always the member nearest the mean:", bool(ok))
Jm = KMeans(2, init=np.array([[0, 5], [5, 5.]]), n_init=1).fit(X8).inertia_
import itertools
Jr = min((((X8[:, None] - X8[list(c)][None]) ** 2).sum(2).min(1).sum(), c) for c in itertools.combinations(range(8), 2))
print("eight customers: J with free centres", Jm, " best two customers as depots", Jr[0], "customers", [i + 1 for i in Jr[1]],
      " increase", round(100 * (Jr[0] / Jm - 1), 1), "per cent")
for c in ([1, 2, 4, 5], [0, 3, 6, 7]):
    P = X8[c]; mu = P.mean(0); d = np.linalg.norm(P - mu, axis=1); k = d.argmin()
    print("  cluster", [i + 1 for i in c], "mean", mu, " nearest member: customer", c[k] + 1, " distance", round(d[k], 4),
          " extra cost n d^2 =", round(len(c) * d[k] ** 2, 4))

print("\n== Q3  an internal score compares only labellings of one space")
Xr = load_wine().data; y = load_wine().target; Xs = StandardScaler().fit_transform(Xr)
lr = KMeans(3, n_init=10, random_state=0).fit(Xr).labels_; ls = KMeans(3, n_init=10, random_state=0).fit(Xs).labels_
for nm, l in (("raw labels", lr), ("standardised labels", ls)):
    print(f"{nm:20s} silhouette in raw space {silhouette_score(Xr, l):.3f}  in standardised space {silhouette_score(Xs, l):.3f}  ARI {adjusted_rand_score(y, l):.3f}")
print("share of total variance of the raw data in proline:", round(Xr[:, 12].var() / Xr.var(0).sum(), 4))

print("\n== Q4  the inertia against K")
print("eight customers, K=1..8:", [round(KMeans(k, n_init=10, random_state=0).fit(X8).inertia_, 2) for k in range(1, 9)])
Xb, _ = make_blobs(600, centers=4, cluster_std=1.5, random_state=0)
tr, te = Xb[:300], Xb[300:]
print("   K  train J   test J (centres from the training half)")
for k in (1, 2, 3, 4, 5, 6, 8, 12, 20, 40, 100):
    km = KMeans(k, n_init=5, random_state=0).fit(tr); print(f"{k:4d} {km.inertia_:9.1f} {-km.score(te):9.1f}")

print("\n== Q5  50 million transactions: mini-batch")
Xl, _ = make_blobs(200_000, n_features=20, centers=20, cluster_std=2.0, random_state=0)
hold = Xl[:20000]; rest = Xl[20000:]
full = KMeans(20, n_init=1, random_state=0).fit(rest)
mb = MiniBatchKMeans(20, random_state=0, n_init=3)
for a in range(0, len(rest), 10000): mb.partial_fit(rest[a:a + 10000])    # one chunk at a time
sub = KMeans(20, n_init=1, random_state=0).fit(rest[::100])             # 1 per cent subsample
for nm, km in (("KMeans on all 180000", full), ("MiniBatch, chunks of 10000", mb), ("KMeans on 1 per cent", sub)):
    print(f"{nm:28s} held-out inertia per sample {-km.score(hold) / len(hold):8.3f}")
print("ARI MiniBatch vs KMeans labels on held-out:", round(adjusted_rand_score(full.predict(hold), mb.predict(hold)), 4))
mb2 = MiniBatchKMeans(20, random_state=1, n_init=3)
for a in range(0, len(rest), 10000): mb2.partial_fit(rest[a:a + 10000])
print("two MiniBatch seeds, held-out J per sample:", round(-mb.score(hold) / len(hold), 3), round(-mb2.score(hold) / len(hold), 3))

print("\n== Q6  a centre that is not a sample")
D = load_digits().data; kd = KMeans(10, n_init=10, random_state=0).fit(D)
Jmean = kd.inertia_; Jsnap = 0.0; far = []
for k in range(10):
    P = D[kd.labels_ == k]; mu = kd.cluster_centers_[k]; j = np.linalg.norm(P - mu, axis=1).argmin()
    Jsnap += ((P - P[j]) ** 2).sum(); far.append(np.linalg.norm(P[j] - mu))
print("digits K=10: J with means", round(Jmean, 1), " with the member nearest each mean", round(Jsnap, 1),
      " increase", round(100 * (Jsnap / Jmean - 1), 1), "per cent")
print("distance from a mean image to its nearest member: min", round(min(far), 2), " max", round(max(far), 2),
      " ; rms distance of members to their centre", round(np.sqrt(Jmean / len(D)), 2))
