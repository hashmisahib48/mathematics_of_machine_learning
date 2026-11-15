# Problem 41C.2: K = 2 on the four blobs is stable and not wrong, but it answers a coarser question
import numpy as np
from math import comb
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import metrics as m
import sys; sys.path.insert(0, '.')
from q_numbers_lib import stability

cent = [(-5, -3), (-3, 4), (4, -4), (5, 3)]
X, y = make_blobs(400, centers=cent, cluster_std=1.5, random_state=3)
mu = np.array([X[y == k].mean(0) for k in range(4)])
print("true blob sizes:", np.bincount(y).tolist(), " blob means:", mu.round(2).tolist())
D = np.linalg.norm(mu[:, None] - mu[None], axis=-1)
print("distances between blob means:", {(i, j): round(float(D[i, j]), 2) for i in range(4) for j in range(i + 1, 4)})
print("spread of a blob (rms distance to its mean): %.2f" % np.sqrt(np.mean([((X[y == k] - mu[k]) ** 2).sum(1).mean() for k in range(4)])))
print("\n K  J(K)     stability  silhouette    CH     DB    ARI")
for K in (1, 2, 3, 4, 5):
    km = KMeans(K, n_init=10, random_state=0).fit(X); l = km.labels_
    if K == 1: print("%2d %8.1f" % (K, km.inertia_)); continue
    print("%2d %8.1f %9.3f %10.3f %8.1f %6.3f %6.3f" % (K, km.inertia_, stability(X, K), m.silhouette_score(X, l), m.calinski_harabasz_score(X, l), m.davies_bouldin_score(X, l), m.adjusted_rand_score(y, l)))
l2 = KMeans(2, n_init=10, random_state=0).fit_predict(X)
print("\ncontingency table, blobs (rows) by K = 2 clusters:"); print(m.cluster.contingency_matrix(y, l2))
t = np.bincount(y); Pt = sum(comb(int(v), 2) for v in t); c = np.bincount(l2); Pc = sum(comb(int(v), 2) for v in c)
P11 = sum(comb(int(v), 2) for v in np.ravel(m.cluster.contingency_matrix(y, l2))); E = Pt * Pc / comb(400, 2)
print("pairs: P11 = %d, pairs together in blobs %d, in clusters %d, E = %.1f, ARI = %.4f (sklearn %.4f)" % (P11, Pt, Pc, E, (P11 - E) / ((Pt + Pc) / 2 - E), m.adjusted_rand_score(y, l2)))
print("homogeneity %.3f  completeness %.3f" % (m.homogeneity_score(y, l2), m.completeness_score(y, l2)))
Z = linkage(X, 'ward'); print("Ward merge heights, last five merges:", Z[-5:, 2].round(1).tolist())
rng = np.random.default_rng(5); same = 0
for r in range(10):                                          # does K = 2 always merge the same pairs?
    ia = rng.choice(400, 320, replace=False); la = KMeans(2, n_init=3, random_state=r).fit_predict(X[ia])
    tab = m.cluster.contingency_matrix(y[ia], la); g = tab.argmax(1); same += int(g[0] == g[1] and g[2] == g[3] and g[0] != g[2])
print("in 10 random 80 per cent subsets, the number in which blobs 0, 1 and blobs 2, 3 form the two clusters: ", same)
