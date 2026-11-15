# Numbers quoted in the model answers to the questions for discussion of Chapter 44
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, ClassicalMDS, LocallyLinearEmbedding, TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
np.set_printoptions(precision=4, suppress=True)

print("== Q1  Isomap on a sphere")
rng = np.random.RandomState(0)
Xs = rng.standard_normal((1000, 3)); Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
cap = Xs[Xs[:, 2] > 0.5]                                  # a cap of angular radius 60 degrees
for name, P in (("whole sphere", Xs), ("cap, 60 deg", cap)):
    iso = Isomap(n_neighbors=10, n_components=2).fit(P); Z = iso.embedding_
    Dg = iso.dist_matrix_; Dz = np.linalg.norm(Z[:, None] - Z[None], axis=2); iu = np.triu_indices(len(P), 1)
    cm = ClassicalMDS(4, metric="precomputed").fit(Dg)
    print(f"{name:13s} n={len(P):4d}  T(10) = {trustworthiness(P, Z, n_neighbors=10):.4f}   correlation of geodesic and map distance {np.corrcoef(Dg[iu], Dz[iu])[0, 1]:.4f}"
          f"   largest geodesic {Dg.max():.3f} (pi = {np.pi:.3f}), map {Dz.max():.3f}")
    print("               top four eigenvalues of B:", cm.eigenvalues_.round(2), " share of the first two:", (cm.eigenvalues_[:2].sum() / cm.eigenvalues_.sum()).round(3))
    print("               relative stress of the map distances against the geodesic ones:", round(float(np.sqrt(((Dg[iu] - Dz[iu]) ** 2).sum() / (Dg[iu] ** 2).sum())), 4))

print("\n== Q2  noisy swiss roll: neighbours that sit on another whorl")
Xn, tn = make_swiss_roll(1000, noise=0.5, random_state=0)
# t of the clean roll is not returned with the noise, so use the clean roll's t (same random_state, same positions)
_, tc = make_swiss_roll(1000, noise=0.0, random_state=0)
print("max |t_noisy - t_clean|:", np.abs(tn - tc).max())
for m in (5, 10, 20, 40):
    nn = NearestNeighbors(n_neighbors=m + 1).fit(Xn); d, idx = nn.kneighbors(Xn); idx = idx[:, 1:]; d = d[:, 1:]
    far = np.abs(tc[idx] - tc[:, None]) > np.pi                  # neighbour more than half a turn away along the sheet
    print(f"m={m:2d}: mean distance to the m-th neighbour {d[:, -1].mean():5.2f};  share of the edges that join different whorls {far.mean():.4f};  share of points with at least one such edge {far.any(1).mean():.3f}")
print("radial spacing of the whorls: 2*pi =", round(2 * np.pi, 3))

print("\n== Q3  LLE: size of the neighbourhood in the map against that in space, along the sheet")
X, t = make_swiss_roll(1000, random_state=0)
def edge_ratio(Z, m=12):
    idx = NearestNeighbors(n_neighbors=m + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    dz = np.linalg.norm(Z[:, None] - Z[idx], axis=2).mean(1); dx = np.linalg.norm(X[:, None] - X[idx], axis=2).mean(1)
    return dz / dx
bins = np.quantile(t, np.linspace(0, 1, 6)); lab = np.digitize(t, bins[1:-1])
for name, est in (("Isomap", Isomap(n_neighbors=10)), ("LLE standard", LocallyLinearEmbedding(n_neighbors=12, random_state=0)),
                  ("LLE Hessian", LocallyLinearEmbedding(n_neighbors=12, method="hessian", random_state=0))):
    r = edge_ratio(est.fit_transform(X)); g = np.array([r[lab == b].mean() for b in range(5)]); g = g / g.mean()
    print(f"{name:13s} relative edge length (map/space) in the five slices from the inner to the outer end of the roll:", g.round(2), " largest/smallest", round(g.max() / g.min(), 2))

