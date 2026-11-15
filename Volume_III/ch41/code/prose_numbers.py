# Numbers quoted in the prose of Chapter 41 that no listing prints.
import itertools
import numpy as np
from math import comb
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (adjusted_rand_score as ari, rand_score,
                             silhouette_score, silhouette_samples,
                             calinski_harabasz_score as chs,
                             davies_bouldin_score as dbs,
                             normalized_mutual_info_score as nmi,
                             v_measure_score, pair_confusion_matrix,
                             adjusted_mutual_info_score as ami)

X = np.array([[1, 1], [2, 3], [3, 1], [4, 3],
              [7, 2], [8, 4], [3, 7], [5, 8]], float)
seg = np.array([0, 0, 0, 1, 1, 1, 2, 2])
clu = np.array([0, 0, 0, 0, 1, 1, 2, 2])
print("toy: pair counts of the segments", sum(
    comb(int(c), 2) for c in np.bincount(seg)), "of the clusters",
    sum(comb(int(c), 2) for c in np.bincount(clu)), "of", comb(8, 2))
# expectation of P11 over all 8! relabellings of the samples
tot = 0
perms = list(itertools.permutations(range(8)))
for p in perms:
    u = clu[list(p)]
    tot += sum(comb(int(c), 2) for c in np.ravel(
        np.histogram2d(seg, u, bins=[3, 3], range=[[0, 3], [0, 3]])[0]))
print("mean P11 over", len(perms), "permutations:", tot / len(perms))
mu = np.array([X[clu == k].mean(0) for k in range(3)])
SSB = sum((clu == k).sum() * ((mu[k] - X.mean(0)) ** 2).sum()
          for k in range(3))
SSW = sum(((X[clu == k] - mu[k]) ** 2).sum() for k in range(3))
print(f"toy SSB {SSB:.3f} SSW {SSW:.3f} CH {SSB / 2 / (SSW / 5):.3f}",
      f"(sklearn {chs(X, clu):.3f}) DB {dbs(X, clu):.4f}")
S = [np.linalg.norm(X[clu == k] - mu[k], axis=1).mean()
     for k in range(3)]
print("toy S_k", np.round(S, 3), "centroid distances",
      np.round([np.linalg.norm(mu[i] - mu[j]) for i, j in
                [(0, 1), (0, 2), (1, 2)]], 3))
two = KMeans(2, n_init=10, random_state=0).fit_predict(X)
print("toy K=2 clusters", two, f"silhouette {silhouette_score(X, two):.3f}",
      f"ARI {ari(seg, two):.3f} AMI {ami(seg, two):.3f}")
print(f"toy K=3 silhouette {silhouette_score(X, clu):.3f} "
      f"ARI {ari(seg, clu):.3f} AMI {ami(seg, clu):.3f}")
print("ARI of [0,0,1,1] against [0,1,0,1]:",
      ari([0, 0, 1, 1], [0, 1, 0, 1]))
print("pair_confusion_matrix(seg, clu):",
      pair_confusion_matrix(seg, clu).tolist())
print("V-measure - NMI (arithmetic):", v_measure_score(seg, clu)
      - nmi(seg, clu))
print("ARI of one cluster against the segments:", ari(seg, np.zeros(8)),
      "; of the segments against themselves:", ari(seg, seg))

Xb, yb = make_blobs(3000, centers=4, cluster_std=1.5, random_state=0)
lab = KMeans(4, n_init=5, random_state=0).fit_predict(Xb)
full = silhouette_score(Xb, lab)
sub = [silhouette_score(Xb, lab, sample_size=300, random_state=s)
       for s in range(5)]
print(f"blobs n=3000: silhouette {full:.4f}; sample_size=300 gives "
      f"{min(sub):.4f} to {max(sub):.4f} over 5 seeds")

# terms of the by-hand examples
N = np.zeros((3, 3), int)
np.add.at(N, (seg, clu), 1)
ni, nj = N.sum(1), N.sum(0)
print("MI terms (i, j, n_ij, n_i., n_.j, term):")
for i, j in zip(*np.nonzero(N)):
    print(i + 1, j + 1, N[i, j], ni[i], nj[j],
          round(N[i, j] / 8 * np.log(8 * N[i, j] / (ni[i] * nj[j])), 5))
print("H(c) terms", np.round(-(ni / 8) * np.log(ni / 8), 5),
      "H(u) terms", np.round(-(nj / 8) * np.log(nj / 8), 5))
from scipy.spatial.distance import cdist
D = cdist(X, X)
print("customer, a, mean to other clusters, b, s")
for p in range(8):
    own = (clu == clu[p]) & (np.arange(8) != p)
    means = [D[p, clu == k].mean() for k in range(3) if k != clu[p]]
    b = min(means)
    a = D[p, own].mean()
    print("ABCDEFGH"[p], round(a, 3), np.round(means, 3), round(b, 3),
          round((b - a) / max(a, b), 3))

print("share of pairs together in 3 classes of 50:",
      round(3 * comb(50, 2) / comb(150, 2), 4))
