# Numbers quoted in the model answers to the questions for discussion (Q1 - Q4) of Chapter 41
import numpy as np
from math import comb
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_blobs
from sklearn import metrics as m
from sklearn.preprocessing import StandardScaler

from q_numbers_lib import stability

print("== Q1  raw against standardised wine")
Xr, y = load_wine(return_X_y=True); Xs = StandardScaler().fit_transform(Xr)
km = lambda Z: KMeans(3, n_init=10, random_state=0).fit_predict(Z)
lr, ls = km(Xr), km(Xs)
print("variance share of proline in the raw data: %.4f" % (Xr.var(0)[-1] / Xr.var(0).sum()))
print("silhouette            raw space  standardised space")
for nm, l in (("raw labels", lr), ("standardised labels", ls), ("cultivars", y)):
    print("%-20s %9.3f %19.3f" % (nm, m.silhouette_score(Xr, l), m.silhouette_score(Xs, l)))
print("ARI against cultivars: raw %.3f  standardised %.3f" % (m.adjusted_rand_score(y, lr), m.adjusted_rand_score(y, ls)))
print("stability (K = 3): raw %.3f  standardised %.3f" % (stability(Xr, 3), stability(Xs, 3)))
print("ARI between the two clusterings: %.3f" % m.adjusted_rand_score(lr, ls))

print("\n== Q2  how much does one customer cost in ARI?")
seg = np.array([0, 0, 0, 1, 1, 1, 2, 2]); clu = np.array([0, 0, 0, 0, 1, 1, 2, 2])
X8 = np.array([[1, 1], [2, 3], [3, 1], [4, 3], [7, 2], [8, 4], [3, 7], [5, 8]], float)
print("ARI of the eight customers: %.4f  homogeneity %.4f  completeness %.4f" % (m.adjusted_rand_score(seg, clu), m.homogeneity_score(seg, clu), m.completeness_score(seg, clu)))
print("silhouette of the segments in the feature space: %.3f ; of the clusters: %.3f" % (m.silhouette_score(X8, seg), m.silhouette_score(X8, clu)))
print("silhouette of each customer under the segments:", m.silhouette_samples(X8, seg).round(2).tolist())
for r in (1, 5, 10, 50):                                   # the same pattern with r times as many customers, one wrong in r
    s2 = np.repeat(seg, r); c2 = np.repeat(clu, r)
    print("each customer copied %2d times (n = %3d), %2d customers in the wrong cluster: ARI %.4f" % (r, 8 * r, r, m.adjusted_rand_score(s2, c2)))
print("a K = 2 clustering [A B C D | E F G H] against the segments: ARI %.4f" % m.adjusted_rand_score(seg, [0, 0, 0, 0, 1, 1, 1, 1]))

print("\n== Q3  the ceiling of the Rand index")
n = 150
for sizes in ([50, 50, 50], [75, 75]):
    lim = 1 - sum(comb(s, 2) for s in sizes) / comb(n, 2)
    t = np.repeat(range(len(sizes)), sizes)
    print("classes", sizes, ": limit 1 - sum C(n_i,2)/C(n,2) = %.4f ; RI of 150 singleton clusters = %.4f" % (lim, m.rand_score(t, np.arange(n))))
rng = np.random.default_rng(0); t3 = np.repeat([0, 1, 2], 50); t2 = np.repeat([0, 1], 75)
for k in (2, 8, 32, 150, 1000):
    r3 = np.mean([m.rand_score(t3, rng.integers(0, k, n)) for _ in range(20)]); r2 = np.mean([m.rand_score(t2, rng.integers(0, k, n)) for _ in range(20)])
    print("%4d random clusters: mean RI %.3f (3 classes of 50)  %.3f (2 classes of 75)" % (k, r3, r2))

print("\n== Q4  choosing K on the four blobs")
B, _ = make_blobs(400, centers=[(-5, -3), (-3, 4), (4, -4), (5, 3)], cluster_std=1.5, random_state=3)
print(" K  stability  silhouette      CH")
for K in range(2, 9):
    l = KMeans(K, n_init=10, random_state=0).fit_predict(B)
    print("%2d %9.3f %11.3f %8.1f" % (K, stability(B, K), m.silhouette_score(B, l), m.calinski_harabasz_score(B, l)))
print("wine, standardised:")
for K in range(2, 7):
    l = KMeans(K, n_init=10, random_state=0).fit_predict(Xs)
    print("%2d %9.3f %11.3f %8.1f" % (K, stability(Xs, K), m.silhouette_score(Xs, l), m.calinski_harabasz_score(Xs, l)))
