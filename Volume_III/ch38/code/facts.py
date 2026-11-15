# Checks of the statements about scikit-learn made in Chapter 38.
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import (DBSCAN, HDBSCAN, OPTICS, MeanShift, estimate_bandwidth,
                             cluster_optics_dbscan)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari, pairwise_distances
from birds import birds, varying
from hdbscan_tree import hdbscan_scratch

warnings.simplefilter("error", DeprecationWarning)
X, y = birds()
TOY = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [1, 2],
                [6, 1], [7, 1], [7, 2], [8, 2], [4, 5]], float)
print("1. eps is inclusive, min_samples counts the point itself")
print("  distance exactly eps:", DBSCAN(eps=1, min_samples=2).fit([[0, 0], [1, 0]]).labels_)
T3 = [[0, 0], [1, 0], [0.5, 0.2]]
print("  3 points, min_samples=3:", DBSCAN(eps=1, min_samples=3).fit(T3).labels_,
      " min_samples=4:", DBSCAN(eps=1, min_samples=4).fit(T3).labels_)

print("2. DBSCAN with min_samples=1 is single linkage cut at eps")
for eps in (0.1, 0.2, 0.3):
    a = DBSCAN(eps=eps, min_samples=1).fit(X).labels_
    b = fcluster(linkage(X, "single"), t=eps, criterion="distance")
    print(f"  eps={eps}: ARI={ari(a, b):.3f}, clusters={a.max() + 1}")

print("3. HDBSCAN: min_samples=None means min_cluster_size")
a = HDBSCAN(min_cluster_size=10, copy=False).fit(X).labels_
b = HDBSCAN(min_cluster_size=10, min_samples=10, copy=False).fit(X).labels_
print("  identical:", np.array_equal(a, b))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    HDBSCAN().fit(X)
print("  default copy:", [type(i.message).__name__ for i in w])
print("  predict exists?  DBSCAN", hasattr(DBSCAN(), "predict"),
      " HDBSCAN", hasattr(HDBSCAN(), "predict"),
      " OPTICS", hasattr(OPTICS(), "predict"), " MeanShift", hasattr(MeanShift(), "predict"))

print("4. HDBSCAN scratch pipeline agrees with the library")
Xv, yv = varying()
ok = []
for name, XX in (("birds", X), ("varying", Xv)):
    for ms, mcs, meth in ((5, 5, "eom"), (5, 10, "leaf"), (3, 15, "eom"), (8, 8, "eom")):
        h = HDBSCAN(min_samples=ms, min_cluster_size=mcs, cluster_selection_method=meth, copy=False).fit(XX)
        t = h._single_linkage_tree_      # private; used for this check only
        Z = np.c_[t["left_node"], t["right_node"], t["value"], t["cluster_size"]]
        r = hdbscan_scratch(XX, ms, mcs, meth, Z=Z)
        r2 = hdbscan_scratch(XX, ms, mcs, meth)
        same = np.array_equal(h.labels_, r["labels"]) and np.allclose(h.probabilities_, r["prob"])
        heights = np.allclose(np.sort(Z[:, 2]), np.sort(r2["Z"][:, 2]))
        ok.append(same and heights)
print("  all 8 configurations identical (labels, probabilities, merge heights):", all(ok))

print("5. HDBSCAN: dbscan_clustering is DBSCAN without border points")
for eps, m in ((0.2, 5), (0.15, 5), (0.3, 8)):
    h = HDBSCAN(min_samples=m, min_cluster_size=m, copy=False).fit(X)
    cut = h.dbscan_clustering(cut_distance=eps, min_cluster_size=m)
    db = DBSCAN(eps=eps, min_samples=m).fit(X)
    core = np.zeros(len(X), bool); core[db.core_sample_indices_] = True
    border = (~core) & (db.labels_ != -1)
    print(f"  eps={eps} m={m}: core points agree (ARI {ari(cut[core], db.labels_[core]):.2f}); "
          f"border points {border.sum()}, noise DBSCAN {(db.labels_ == -1).sum()}, noise of the cut {(cut == -1).sum()}")

print("6. OPTICS: core distance, extraction rule")
o = OPTICS(min_samples=5).fit(X)
c5 = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, -1]
print("  core_distances_ == distance to 5th nearest (self first):", np.allclose(o.core_distances_, c5))
print("  infinite reachabilities:", int(np.isinf(o.reachability_).sum()), " first point:", o.ordering_[0])
for eps in (0.15, 0.2, 0.3):
    lo = cluster_optics_dbscan(reachability=o.reachability_, core_distances=o.core_distances_, ordering=o.ordering_, eps=eps)
    ld = DBSCAN(eps=eps, min_samples=5).fit(X).labels_
    far, near = o.reachability_ > eps, o.core_distances_ <= eps
    lab = np.zeros(len(X), int)
    lab[o.ordering_] = np.cumsum((far & near)[o.ordering_]) - 1
    lab[far & ~near] = -1
    print(f"  eps={eps}: matches the rule {np.array_equal(lab, lo)}; differs from DBSCAN in {(lo != ld).sum()} points")
op = OPTICS(min_samples=4, cluster_method="dbscan", eps=1.5).fit(TOY)
print("  toy: OPTICS labels", op.labels_, " DBSCAN", DBSCAN(eps=1.5, min_samples=4).fit(TOY).labels_)

print("7. MeanShift: bandwidth estimate and a from-scratch run")
bw = estimate_bandwidth(X, quantile=0.3)
k = int(len(X) * 0.3)
mine = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1].mean()
print(f"  estimate_bandwidth {bw:.4f}, formula {mine:.4f}, k = {k}")


def my_meanshift(X, h):
    modes = {}
    for s in X:
        x = s.copy()
        while True:
            w = np.linalg.norm(X - x, axis=1) <= h
            nx = X[w].mean(axis=0)
            done = np.linalg.norm(nx - x) <= 1e-3 * h
            x = nx
            if done:
                break
        modes[tuple(x)] = int(w.sum())
    order = sorted(modes.items(), key=lambda t: (t[1], t[0]), reverse=True)
    C = np.array([c for c, _ in order])
    keep = np.ones(len(C), bool)
    for i, c in enumerate(C):
        if keep[i]:
            keep[np.linalg.norm(C - c, axis=1) <= h] = False
            keep[i] = True
    C = C[keep]
    return C, pairwise_distances(X, C).argmin(1)


for h in (0.5, 0.9, 1.5):
    C, lab = my_meanshift(X, h)
    ms = MeanShift(bandwidth=h).fit(X)
    print(f"  h={h}: scratch {len(C)} modes, MeanShift {len(ms.cluster_centers_)}, labels equal "
          f"{np.array_equal(lab, ms.labels_)}")
from sklearn.cluster import get_bin_seeds
print("  seeds: all points", len(X), " bin_seeding at h=0.9:", len(get_bin_seeds(X, 0.9, 1)))

print("8. Other arguments")
w = np.ones(len(TOY)); w[9] = 4
print("  sample_weight 4 on the stray, min_samples=4:", DBSCAN(eps=1.5, min_samples=4).fit(TOY, sample_weight=w).labels_)
D = pairwise_distances(X)
print("  precomputed distances give the same labels:",
      np.array_equal(DBSCAN(eps=0.2, min_samples=5, metric="precomputed").fit(D).labels_,
                     DBSCAN(eps=0.2, min_samples=5).fit(X).labels_))
rng = np.random.default_rng(0)
one = rng.normal(size=(150, 2))
a = HDBSCAN(min_cluster_size=10, copy=False).fit(one)
b = HDBSCAN(min_cluster_size=10, allow_single_cluster=True, copy=False).fit(one)
print(f"  one Gaussian blob: default {a.labels_.max() + 1} clusters ({(a.labels_ == -1).sum()} noise); "
      f"allow_single_cluster {b.labels_.max() + 1} cluster ({(b.labels_ == -1).sum()} noise)")
h1 = HDBSCAN(min_cluster_size=3, copy=False).fit(X)
print(f"  min_cluster_size=3: {h1.labels_.max() + 1} clusters, {(h1.labels_ == -1).sum()} noise")
for e in (0.05, 0.15):
    try:
        h2 = HDBSCAN(min_cluster_size=3, cluster_selection_epsilon=e, copy=False).fit(X)
        print(f"  cluster_selection_epsilon={e}: {h2.labels_.max() + 1} clusters, {(h2.labels_ == -1).sum()} noise")
    except Exception as exc:
        print(f"  cluster_selection_epsilon={e}: {type(exc).__name__}: {str(exc)[:60]}")
hc = HDBSCAN(min_cluster_size=10, store_centers="both", copy=False).fit(X)
print("  store_centers='both':", hc.centroids_.shape, hc.medoids_.shape)
o2 = OPTICS(min_samples=0.05).fit(X)
print("  OPTICS min_samples=0.05 means", max(2, int(0.05 * len(X))), "of", len(X), "samples; clusters", o2.labels_.max() + 1)
ox = OPTICS(min_samples=8, xi=0.1, min_cluster_size=20).fit(Xv)
print("  xi hierarchy: shape", ox.cluster_hierarchy_.shape, "labels_ clusters", ox.labels_.max() + 1)
