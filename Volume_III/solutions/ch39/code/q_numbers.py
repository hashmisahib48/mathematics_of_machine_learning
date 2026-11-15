# Numbers quoted in the model answers of Chapter 39. Run: python3 q_numbers.py
import sys, os, itertools, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch39/code')
warnings.simplefilter('ignore', UserWarning); warnings.simplefilter('ignore', RuntimeWarning)
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import spectral_embedding
from data import rings, toy_adjacency

# ---- Question 1: ranking of the splits of the six points by cut and by Ncut
W = toy_adjacency(); d = W.sum(1); rows = []
for r in range(1, 6):
    for A in itertools.combinations(range(6), r):
        if 0 in A:
            B = [i for i in range(6) if i not in A]; cut = W[np.ix_(A, B)].sum()
            rows.append((cut, cut * (1 / d[list(A)].sum() + 1 / d[B].sum()), [a + 1 for a in A]))
print("Q1: six points, %d splits" % len(rows))
byc = sorted(rows, key=lambda t: (t[0], t[1])); byn = sorted(rows, key=lambda t: t[1])
print("  by cut: smallest", [(float(c), round(float(n), 3), A) for c, n, A in byc[:7]])
print("  splits with cut 2:", sum(1 for t in rows if t[0] == 2), " of which a single point (side of size 1):", sum(1 for t in rows if t[0] == 2 and (len(t[2]) == 1 or len(t[2]) == 5)))
print("  by Ncut: smallest", [(float(c), round(float(n), 3), A) for c, n, A in byn[:5]])
for c, n, A in rows:
    if A == [1]: print("  point 1 alone: cut %g Ncut %.4f, rank by Ncut %d of %d" % (c, n, 1 + sorted(t[1] for t in rows).index(n), len(rows)))
# a pendant point: the path 1-2-3-4 has the same cut 1 for three splits
print("  path of four nodes: all three splits have cut 1; Ncut 1.200, 0.667, 1.200 (39A.1)")
print("  number of splits of n points:", 2 ** 5 - 1, "for n = 6;", 2 ** 299 - 1 > 10 ** 80, "for n = 300 more than 10^80:", len(str(2 ** 299 - 1)), "digits")

# ---- Question 2: what the embedding does to the rings
X, y = rings(); g = 30.0
Wr = rbf_kernel(X, gamma=g); np.fill_diagonal(Wr, 0)
E = spectral_embedding(Wr, n_components=2, drop_first=False, random_state=0)
u = E[:, 1]
print("\nQ2: embedding of the rings (gamma = 30), coordinate u2")
print("  outer ring: u2 from %.5f to %.5f; inner ring: from %.5f to %.5f" % (u[y == 0].min(), u[y == 0].max(), u[y == 1].min(), u[y == 1].max()))
print("  gap between the rings in u2: %.5f; width of each ring: %.5f, %.5f" % (abs(u[y == 0].mean() - u[y == 1].mean()), np.ptp(u[y == 0]), np.ptp(u[y == 1])))
print("  k-means on raw points ARI %.3f; k-means on u2 ARI %.3f" % (ari(y, KMeans(2, n_init=10, random_state=0).fit_predict(X)), ari(y, KMeans(2, n_init=10, random_state=0).fit_predict(E))))
dd = ((X[:, None] - X[None]) ** 2).sum(-1)
same = y[:, None] == y[None]
print("  mean affinity within a ring %.4f, between rings %.2e; largest between-ring affinity %.2e" % (Wr[same & ~np.eye(300, dtype=bool)].mean(), Wr[~same].mean(), Wr[~same].max()))
print("  smallest distance between the rings %.3f" % np.sqrt(dd[~same].min()))
# a bad affinity: gamma so small that rings are connected
Wb = rbf_kernel(X, gamma=1.0); np.fill_diagonal(Wb, 0)
Eb = spectral_embedding(Wb, n_components=2, drop_first=False, random_state=0)
print("  gamma = 1: ARI of k-means on embedding %.3f; mean affinity within %.3f between %.3f" % (ari(y, KMeans(2, n_init=10, random_state=0).fit_predict(Eb)), Wb[same & ~np.eye(300, dtype=bool)].mean(), Wb[~same].mean()))

# ---- Question 3: units
print("\nQ3: unit of length")
for name, scale in (("kilometres", 1.0), ("metres", 1000.0)):
    Xs = X * scale
    for gam in (1.0,):
        Wm = rbf_kernel(Xs, gamma=gam); np.fill_diagonal(Wm, 0)
        comp = connected_components(Wm > 1e-8)[0]
        lab = SpectralClustering(2, gamma=gam, random_state=0).fit_predict(Xs)
        print("  %s, gamma = 1: weights above 1e-8 give %d graph pieces; ARI %.3f" % (name, comp, ari(y, lab)))
from sklearn.neighbors import NearestNeighbors
for name, scale in (("kilometres", 1.0), ("metres", 1000.0)):
    Xs = X * scale; dist = NearestNeighbors(n_neighbors=10).fit(Xs).kneighbors(Xs)[0][:, -1]
    sig = dist.mean(); gam = 1 / (2 * sig ** 2)
    lab = SpectralClustering(2, gamma=gam, random_state=0).fit_predict(Xs)
    print("  %s: mean distance to the 10th neighbour (self first) %.4g, gamma = 1/(2 sigma^2) = %.4g, ARI %.3f" % (name, sig, gam, ari(y, lab)))
print("  conversion: gamma in 1/m^2 = gamma in 1/km^2 / 1e6 = %.3g" % (19.6 / 1e6))

# ---- Question 4: preference against K
Xb, yb = make_blobs(90, centers=3, cluster_std=0.9, random_state=4)
Sb = -((Xb[:, None] - Xb[None]) ** 2).sum(-1)
print("\nQ4: 90 points in three groups, median of all similarities %.3f" % np.median(Sb))
cnt = {}
for p in -np.logspace(-1, 3.5, 91):
    ap = AffinityPropagation(affinity='precomputed', preference=p, random_state=0, max_iter=400).fit(Sb)
    cnt[p] = (len(ap.cluster_centers_indices_), ari(yb, ap.labels_))
three = [p for p, (k, a) in cnt.items() if k == 3]
print("  preferences (grid of 91 values from -0.1 to -3162) with exactly 3 exemplars: from %.1f to %.1f; |p| range ratio %.1f" % (max(three), min(three), min(three) / max(three)))
print("  exemplars at p = median: ", AffinityPropagation(affinity='precomputed', random_state=0).fit(Sb).cluster_centers_indices_.size)
ks = sorted(set(k for k, a in cnt.values())); print("  numbers of exemplars met on the grid:", ks[:6], "...", ks[-3:])
print("  smallest p of the grid (-3162): exemplars", cnt[min(cnt)][0], " ; p = -120: ", AffinityPropagation(affinity='precomputed', preference=-120, random_state=0).fit(Sb).cluster_centers_indices_.size)
print("  spectral clustering K = 3 on these points (gamma 0.3): ARI %.3f" % ari(yb, SpectralClustering(3, gamma=0.3, random_state=0).fit_predict(Xb)))
sc = {K: round(float(ari(yb, SpectralClustering(K, gamma=0.3, random_state=0).fit_predict(Xb))), 3) for K in (2, 3, 4, 5)}
print("  spectral K -> ARI:", sc)
