# Numbers quoted in the model answers to the questions for discussion of Chapter 37.
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering as AC, Birch
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("== Q1  the factor n_A n_B / (n_A + n_B)")
dist = 4.0
for nA, nB in ((1, 1), (1, 100), (100, 100), (1, 10)):
    f = nA * nB / (nA + nB); print(f"  n_A={nA:3d} n_B={nB:3d}: factor {f:8.4f}   Delta = {f*dist**2:9.3f}   height sqrt(2 Delta) = {np.sqrt(2*f*dist**2):.3f}")
X7 = np.array([[3,3],[1,2],[0,2],[7,4],[11,2],[5,12],[8,9]], float)
Z = linkage(X7, 'ward'); print("  finch Ward heights", np.round(Z[:, 2], 3).tolist(), " Delta = h^2/2:", np.round(Z[:, 2]**2/2, 2).tolist())
# an outlier next to a big cluster against a pair of outliers
rng = np.random.default_rng(0); bulk = rng.normal(0, 1, (100, 2)); out = np.array([[8., 0.], [9., 1.]])
Xo = np.r_[bulk, out]; Zo = linkage(Xo, 'ward'); print("  100 bulk points + two outliers 1.41 apart: last five Ward heights", np.round(Zo[-5:, 2], 2).tolist())
lab = fcluster(Zo, 3, 'maxclust'); print("  3 clusters, sizes", sorted(np.bincount(lab)[1:].tolist()))
print("  all-single and Ward K=2 sizes of bulk + two outliers: single", sorted(np.bincount(fcluster(linkage(Xo,'single'),2,'maxclust'))[1:]), " ward", sorted(np.bincount(fcluster(Zo,2,'maxclust'))[1:]))

print("\n== Q2  a bridge of five samples (two groups of 40, sd 0.5, centres 4 apart; bridge at x = 1, 1.5, 2, 2.5, 3)")
def bridge_data(seed):
    rng = np.random.default_rng(seed); A = rng.normal([0, 0], 0.5, (40, 2)); B = rng.normal([4, 0], 0.5, (40, 2))
    return np.r_[A, B, np.c_[np.linspace(1, 3, 5), rng.normal(0, 0.08, 5)]]
Xb = bridge_data(0)
for L in ('single', 'complete', 'ward'):
    z = linkage(Xb, L); l2 = fcluster(z, 2, 'maxclust')
    print(f"  seed 0 {L:9s} K=2 sizes {sorted(np.bincount(l2)[1:].tolist())}; bridge samples in the cluster of group A: {int((l2[80:]==l2[0]).sum())}, group B: {int((l2[80:]==l2[40]).sum())}; last heights {np.round(z[-3:,2],2).tolist()}")
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
T = minimum_spanning_tree(squareform(pdist(Xb))).toarray(); w = np.sort(T[T > 0])[::-1]
print("  seed 0: five longest edges of the minimum spanning tree:", np.round(w[:5], 3).tolist())
Dm0 = squareform(pdist(Xb)); print("  seed 0: distances between consecutive bridge samples:", np.round([Dm0[i, i+1] for i in range(80, 84)], 3).tolist())
ei, ej = np.nonzero(T); k = np.argmax(T[ei, ej]); print("  longest tree edge joins samples", int(ei[k]), "and", int(ej[k]), "(bridge samples are 80 to 84):", bool(ei[k] >= 80 or ej[k] >= 80))
print("  bridge edges among the five longest tree edges:", sum(1 for q in np.argsort(-T[ei, ej])[:5] if ei[q] >= 80 or ej[q] >= 80))
res = {L: [] for L in ('single', 'complete', 'ward')}
for sd in range(20):
    Xs_ = bridge_data(sd)
    for L in res:
        l2 = fcluster(linkage(Xs_, L), 2, 'maxclust'); res[L].append((min(np.bincount(l2)[1:]), int((l2[80:]==l2[0]).sum())))
for L, v in res.items():
    print(f"  20 seeds, {L:9s}: smaller cluster has 1 sample in {sum(m==1 for m,_ in v)} runs; groups cleanly separated (smaller cluster 40 to 45) in {sum(40<=m<=45 for m,_ in v)} runs; bridge samples left with group A: {sorted(set(b for _,b in v))}")

print("\n== Q3  distance_threshold = 5")
Xw = load_wine().data; Xs = StandardScaler().fit_transform(Xw)
for nme, M in (("raw", Xw), ("standardised", Xs), ("standardised, x10", Xs * 10), ("standardised, /10", Xs / 10)):
    k = {L: int(AC(n_clusters=None, distance_threshold=5, linkage=L).fit(M).n_clusters_) for L in ('ward', 'complete', 'average', 'single')}
    print(f"  wine {nme:20s}", k)
print("  finches Ward heights", np.round(Z[:, 2], 2).tolist())
for s, nme in ((1, "cm"), (10, "mm (x10)"), (0.1, "x0.1")):
    print(f"  finches with all coordinates x{s}: clusters at t=5:", int(AC(n_clusters=None, distance_threshold=5).fit(X7 * s).n_clusters_))
print("  wine standardised: Ward heights near the top", np.round(AC(n_clusters=None, distance_threshold=0, compute_full_tree=True).fit(Xs).distances_[-6:], 2).tolist())

print("\n== Q4  labelling a new sample")
Xtr, Xte = train_test_split(Xs, test_size=0.25, random_state=0)
ac = AC(n_clusters=3).fit(Xtr); lab = ac.labels_
cen = np.array([Xtr[lab == k].mean(0) for k in range(3)]); sz = np.bincount(lab)
nc = ((Xte[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
wd = np.array([sz[k] / (sz[k] + 1) * ((Xte - cen[k]) ** 2).sum(1) for k in range(3)]).argmin(0)
kn = KNeighborsClassifier(5).fit(Xtr, lab).predict(Xte)
ref = AC(n_clusters=3).fit(np.r_[Xtr, Xte]).labels_[len(Xtr):]
print("  train", len(Xtr), "test", len(Xte), " cluster sizes", sz.tolist())
print("  ARI with the labels of a refit on train+test: nearest centroid %.3f   Ward cost rule %.3f   5-NN %.3f" % (ari(ref, nc), ari(ref, wd), ari(ref, kn)))
print("  nearest centroid and Ward cost rule differ on", int((nc != wd).sum()), "of", len(Xte), "test wines")
print("  the refit changes the labels of training wines? ARI between old and refit labels of the 133 training wines: %.3f" % ari(lab, AC(n_clusters=3).fit(np.r_[Xtr, Xte]).labels_[:len(Xtr)]))
b = Birch(threshold=1.5, n_clusters=3).fit(Xtr)
near = b.subcluster_labels_[((Xte[:, None] - b.subcluster_centers_[None]) ** 2).sum(-1).argmin(1)]
print("  Birch subclusters", len(b.subcluster_centers_), " predict(test) equals the label of the nearest subcluster centre:", bool((near == b.predict(Xte)).all()),
      " ARI with refit %.3f" % ari(ref, b.predict(Xte)))
