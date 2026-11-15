# Numbers quoted in the prose of Chapter 44 (not shown as a listing).
import warnings
import numpy as np
from scipy.linalg import eigh
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import (ClassicalMDS, Isomap, MDS, TSNE,
    LocallyLinearEmbedding, SpectralEmbedding, trustworthiness)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
n = len(Xtr)
cm = ClassicalMDS(2).fit(Xtr); pc = PCA(2).fit(Xtr)
print("1 eigenvalues_/explained_variance_:",
      (cm.eigenvalues_ / pc.explained_variance_).round(6), "n-1 =", n - 1)

R, t = make_swiss_roll(600, random_state=0)
iso = Isomap(n_neighbors=10).fit(R)
cg = ClassicalMDS(2, metric="precomputed").fit(iso.dist_matrix_)
print("2 Isomap vs ClassicalMDS on dist_matrix_: max |diff| =",
      f"{np.abs(iso.embedding_ - cg.embedding_).max():.1e}")
isoFW = Isomap(n_neighbors=10, path_method="FW").fit(R)
print("2b path_method D vs FW: max |diff| =",
      f"{np.abs(iso.dist_matrix_ - isoFW.dist_matrix_).max():.1e}")

lle = LocallyLinearEmbedding(n_neighbors=12, eigen_solver="dense",
                             random_state=0).fit(R)
Z = lle.embedding_
print("3 LLE: Z^T Z = I:", np.allclose(Z.T @ Z, np.eye(2)),
      " column sums:", np.abs(Z.sum(axis=0)).max().round(8))
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph
W = barycenter_kneighbors_graph(lle.nbrs_, 12, reg=1e-3).toarray()
M = (np.eye(600) - W).T @ (np.eye(600) - W)
ev = np.linalg.eigvalsh(M)
print("3 LLE: smallest eigenvalues of M:", ev[:4].round(8))
print("3 LLE: reconstruction_error_ =", f"{lle.reconstruction_error_:.6e}",
      " sum of eigenvalues 2 and 3 =", f"{ev[1:3].sum():.6e}")
print("3 LLE: tr(Z^T M Z) =", f"{np.trace(Z.T @ M @ Z):.6e}")

se = SpectralEmbedding(n_components=2, n_neighbors=10,
                       random_state=0).fit(R)
Wm = se.affinity_matrix_.toarray()
print('4 Laplacian: diagonal of the affinity matrix:', Wm.diagonal()[:3])
np.fill_diagonal(Wm, 0.0)                  # the Laplacian ignores it
deg = Wm.sum(axis=1)
L = np.diag(deg) - Wm
w, V = eigh(L, np.diag(deg))
print("4 Laplacian: smallest generalised eigenvalues:", w[:4].round(6))
Zs = se.embedding_
print("4 Laplacian: Z^T D Z =\n", (Zs.T @ np.diag(deg) @ Zs).round(6))
print("4 Laplacian: n_neighbors_ (default n/10) for n = 600:",
      SpectralEmbedding(random_state=0).fit(R).n_neighbors_)
print("4 Laplacian: affinity values:", np.unique(Wm.round(3)))
print("4 Laplacian: gamma_ of rbf =",
      SpectralEmbedding(affinity="rbf", random_state=0).fit(Xtr[:200]
      ).gamma_, "= 1/64 =", 1 / 64)

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    Isomap(n_neighbors=2).fit(Xtr[:300])
print("5 Isomap n_neighbors=2 on 300 digits:",
      [str(w.message)[:75] for w in rec if w.category is UserWarning])
print("5 Isomap.transform of the test images:",
      Isomap(n_neighbors=12).fit(Xtr).transform(Xte).shape)

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    MDS(2, init="classical_mds", n_init=4, metric="euclidean",
        random_state=0).fit(R[:50])
print("6 MDS init='classical_mds', n_init=4:",
      [str(w.message) for w in rec])

def trust(X, Z, k):                                # Eq. 44.9-1
    D = pairwise_distances(X); np.fill_diagonal(D, np.inf)
    rank = np.argsort(np.argsort(D, axis=1), axis=1) + 1
    De = pairwise_distances(Z); np.fill_diagonal(De, np.inf)
    nb = np.argsort(De, axis=1)[:, :k]
    r = np.take_along_axis(rank, nb, axis=1)
    pen = np.where(r > k, r - k, 0).sum()
    m = len(X)
    return 1 - 2 * pen / (m * k * (2 * m - 3 * k - 1))
Zp = PCA(2).fit_transform(Xtr[:300])
print("7 trustworthiness: ours", round(trust(Xtr[:300], Zp, 10), 6),
      " scikit-learn",
      round(trustworthiness(Xtr[:300], Zp, n_neighbors=10), 6))

D150 = pairwise_distances(X[:150] / 16)
for metric in (True, False):
    m = MDS(2, metric_mds=metric, init="classical_mds", n_init=1,
            metric="precomputed", normalized_stress="auto",
            random_state=0).fit(D150)
    print(f"8 MDS metric_mds={metric}: stress_ = {m.stress_:.4f}"
          f" after {m.n_iter_} iterations")
ex = TSNE(perplexity=30, init="pca", method="exact",
          random_state=0).fit(Xtr[:500])
bh = TSNE(perplexity=30, init="pca", random_state=0).fit(Xtr[:500])
print("9 t-SNE n=500 KL: exact", round(ex.kl_divergence_, 4),
      " barnes_hut", round(bh.kl_divergence_, 4))
print("9 t-SNE n_iter_ =", bh.n_iter_, " learning_rate_ =",
      bh.learning_rate_)
try:
    LocallyLinearEmbedding(n_neighbors=5, method="hessian").fit(R)
except ValueError as e:
    print("10 hessian with m = 5:", str(e)[:70])
print("11 PCA of the swiss roll, variance shares:",
      PCA(3).fit(R).explained_variance_ratio_.round(3))
mds_d = MDS(2, init="classical_mds", n_init=1, metric="euclidean",
            normalized_stress=True, random_state=0).fit(Xtr)
print("12 digits SMACOF Stress-1 =", round(mds_d.stress_, 4),
      " iterations", mds_d.n_iter_)
