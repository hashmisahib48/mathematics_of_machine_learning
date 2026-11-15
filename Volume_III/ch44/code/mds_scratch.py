# Classical MDS and Isomap from scratch.
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph


def classical_mds(D, k=2):
    """Eqs. 44.3-2 and 44.3-3 for a distance matrix D."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J                 # double centring
    w, V = np.linalg.eigh(B)
    w, V = w[::-1][:k], V[:, ::-1][:, :k]       # largest first
    big = np.abs(V).argmax(axis=0)              # sign rule of
    V = V * np.sign(V[big, np.arange(k)])       # svd_flip
    return V * np.sqrt(w), w


class MyIsomap:
    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X):
        A = kneighbors_graph(X, self.n_neighbors, mode="distance")
        self.dist_matrix_ = shortest_path(    # Eq. 44.3-4
            A, method="D", directed=False)
        self.embedding_, self.eigenvalues_ = classical_mds(
            self.dist_matrix_, self.n_components)
        return self.embedding_


# --- the tests
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import ClassicalMDS, Isomap
from sklearn.metrics import pairwise_distances

X, t = make_swiss_roll(500, random_state=0)
D = pairwise_distances(X)
Z, w = classical_mds(D)
ref = ClassicalMDS(2, metric="precomputed").fit(D)
print("ClassicalMDS: max |dZ| =",
      f"{np.abs(Z - ref.embedding_).max():.1e}",
      " eigenvalues", w.round(2), ref.eigenvalues_.round(2))
mine = MyIsomap(10, 2)
Zi = mine.fit_transform(X)
iso = Isomap(n_neighbors=10, n_components=2).fit(X)
print("Isomap: max |dZ| =",
      f"{np.abs(Zi - iso.embedding_).max():.1e}", " max |dD| =",
      f"{np.abs(mine.dist_matrix_ - iso.dist_matrix_).max():.1e}")
assert np.allclose(Z, ref.embedding_)
assert np.allclose(mine.dist_matrix_, iso.dist_matrix_)
assert np.allclose(Zi, iso.embedding_, atol=1e-6)
