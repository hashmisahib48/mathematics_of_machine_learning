# Checks of the statements about the source code of scikit-learn 1.9.1.
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import (AffinityPropagation, SpectralBiclustering,
                             SpectralClustering, SpectralCoclustering)
from sklearn.manifold import spectral_embedding
from sklearn.metrics import adjusted_rand_score, consensus_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_checkerboard
from data import rings, toy_adjacency


def same_partition(a, b):
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


X, y = rings()
# 1. the diagonal of the affinity matrix is ignored
W = rbf_kernel(X, gamma=30.0)
W0 = W.copy(); np.fill_diagonal(W0, 0.0)
e1 = spectral_embedding(W, n_components=2, drop_first=False, random_state=0)
e0 = spectral_embedding(W0, n_components=2, drop_first=False, random_state=0)
print("1. diagonal ignored: max |difference| =", np.abs(e1 - e0).max())

# 2. the embedding is D^(-1/2) times the eigenvectors of L_sym
d = W0.sum(axis=1)
Ls = np.eye(len(X)) - W0 / np.sqrt(np.outer(d, d))
lam, V = np.linalg.eigh(Ls)
U = V[:, :2] / np.sqrt(d)[:, None]
err = [min(np.abs(e0[:, j] - s * U[:, j]).max() for s in (1, -1)) for j in (0, 1)]
print("2. embedding = D^-1/2 V: max error per column", np.round(err, 7))
print("   first column constant:", np.ptp(e0[:, 0]) < 1e-9,
      "; row norms not 1:", np.linalg.norm(e0, axis=1)[:3].round(4))

# 3. nearest_neighbors affinity
sc = SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=10,
                        random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    sc.fit(X)
A = sc.affinity_matrix_
vals = np.unique(A.data)
print("3. affinity_matrix_ values:", vals, " symmetric:",
      abs(A - A.T).max() == 0, " diagonal all 1:", bool((A.diagonal() == 1).all()))
nz = np.asarray((A > 0).sum(axis=1)).ravel()
print("   entries per row (self included) min/median/max:",
      nz.min(), int(np.median(nz)), nz.max())

# 4. precomputed affinity on the six-point toy
t = SpectralClustering(2, affinity="precomputed", random_state=0)
print("4. toy labels:", t.fit_predict(toy_adjacency()).tolist())
print("   has predict:", hasattr(SpectralClustering, "predict"),
      "; AffinityPropagation has predict:", hasattr(AffinityPropagation, "predict"))

# 5. default preference of affinity propagation: the median of all n^2 entries
pts = np.array([0.0, 1.0, 3.0]); S3 = -(pts[:, None] - pts[None]) ** 2
off = S3[~np.eye(3, dtype=bool)]
print("5. median of all 9 entries:", np.median(S3), "; of the 6 off-diagonal:",
      np.median(off))
Xr = rings(100)[0]
Sr = -((Xr[:, None] - Xr[None]) ** 2).sum(-1)
offr = Sr[~np.eye(100, dtype=bool)]
print("   100 ring points: median all", round(float(np.median(Sr)), 4),
      "off-diagonal", round(float(np.median(offr)), 4))

# 6. spectral co-clustering: the construction of the source in NumPy
rng = np.random.default_rng(0)
M = rng.poisson(1.0, (30, 20)).astype(float)
M[:15, :10] += rng.poisson(4.0, (15, 10)); M[15:, 10:] += rng.poisson(4.0, (15, 10))
co = SpectralCoclustering(2, random_state=0).fit(M)
r = 1 / np.sqrt(M.sum(1)); c = 1 / np.sqrt(M.sum(0))
An = r[:, None] * M * c
u, s, vt = np.linalg.svd(An)
n_sv = 1 + int(np.ceil(np.log2(2)))
z = np.vstack([r[:, None] * u[:, 1:n_sv], c[:, None] * vt.T[:, 1:n_sv]])
from sklearn.cluster import KMeans
lab = KMeans(2, init="k-means++", n_init=10, random_state=0).fit(z).labels_
print("6. co-clustering rebuilt in NumPy: same rows",
      same_partition(lab[:30], co.row_labels_), ", same columns",
      same_partition(lab[30:], co.column_labels_), "; n_sv for K=2,3,4,5:",
      [1 + int(np.ceil(np.log2(k))) for k in (2, 3, 4, 5)])
print("   singular values of the normalised matrix:", s[:4].round(4))

# 7. biclustering shapes and consensus score
data, rows, cols = make_checkerboard((60, 60), (3, 3), noise=3, shuffle=False,
                                      random_state=0)
sb = SpectralBiclustering((3, 3), method="log", random_state=0).fit(data)
print("7. rows_ shape", sb.rows_.shape, "columns_ shape", sb.columns_.shape,
      "; consensus score", round(consensus_score(sb.biclusters_, (rows, cols)), 3))
print("   a perfect match gives", consensus_score((rows, cols), (rows, cols)))

# 8. undamped affinity propagation can oscillate (points 0, 1, 4; p = -6)
pts = np.array([0.0, 1.0, 4.0]); S4 = -(pts[:, None] - pts[None]) ** 2
np.fill_diagonal(S4, -6.0)
Am = np.zeros((3, 3)); crit = []
for it in range(6):
    Rm = np.zeros((3, 3))
    for i_ in range(3):
        for k_ in range(3):
            Rm[i_, k_] = S4[i_, k_] - max(Am[i_, j] + S4[i_, j] for j in range(3) if j != k_)
    Am = np.zeros((3, 3))
    for k_ in range(3):
        for i_ in range(3):
            pos = [max(0, Rm[j, k_]) for j in range(3) if j not in (i_, k_)]
            Am[i_, k_] = sum(pos) if i_ == k_ else min(0, Rm[k_, k_] + sum(pos))
    crit.append(float(Rm[0, 1]))
print("8. undamped r(1,2) in rounds 1 to 6:", crit)

# 9. a scale for gamma: the distance to the 10th neighbour on the two rings
from sklearn.neighbors import NearestNeighbors
dist = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)[0][:, -1]
print("9. mean distance to the 10th neighbour (self first, so 9th other):",
      round(float(dist.mean()), 3), " 1/(2 s^2) =", round(float(1 / (2 * dist.mean() ** 2)), 1))
