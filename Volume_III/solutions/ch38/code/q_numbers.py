# Numbers quoted in the model answers to the questions for discussion of Chapter 38.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ch38/code'))
import numpy as np
from birds import birds, varying
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, OPTICS, cluster_optics_dbscan
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from hdbscan_tree import hdbscan_scratch

print("== Q1  border points and the order of the data")
xc = np.r_[np.arange(0, .5, .1), 0.9, np.arange(1.4, 1.9, .1)]
print("  distances of 0.9 to the core samples 0.4 and 1.4:", repr(float(0.9 - xc[4])), repr(float(xc[6] - 0.9)))
for nm, order in (("given order", np.arange(11)), ("reversed", np.arange(11)[::-1])):
    l = DBSCAN(eps=.5, min_samples=5).fit(xc[order, None]).labels_; i9 = int(np.flatnonzero(order == 5)[0])
    print(f"  {nm:12s}: label of 0.9 = {l[i9]}, label of 0.4 = {l[int(np.flatnonzero(order == 4)[0])]}, label of 1.4 = {l[int(np.flatnonzero(order == 6)[0])]}")
rng = np.random.default_rng(0); cnt = 0
for _ in range(200):
    o = rng.permutation(11); l = DBSCAN(eps=.5, min_samples=5).fit(xc[o, None]).labels_
    cnt += l[int(np.flatnonzero(o == 5)[0])] == l[int(np.flatnonzero(o == 4)[0])]
print("  200 random orders: 0.9 goes with the left group in", cnt, "orders and with the right group in", 200 - cnt)
X, y = birds()
for eps, m in ((0.2, 5), (0.2, 12)):
    d = DBSCAN(eps=eps, min_samples=m).fit(X); cs = np.zeros(len(X), bool); cs[d.core_sample_indices_] = True
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); amb = 0
    for i in np.flatnonzero(~cs & (d.labels_ >= 0)):
        near = set(d.labels_[(D[i] <= eps) & cs])
        amb += len(near) > 1
    ch = 0; pd = 0; ref = d.labels_
    for _ in range(100):
        o = rng.permutation(len(X)); l = DBSCAN(eps=eps, min_samples=m).fit(X[o]).labels_; back = np.empty_like(l); back[o] = l
        ch += int(((ref == -1) != (back == -1)).sum() > 0); pd += int(ari(ref, back) < 1)
        cs2 = np.zeros(len(X), bool); cs2[o[DBSCAN(eps=eps, min_samples=m).fit(X[o]).core_sample_indices_]] = True; assert (cs2 == cs).all()
    print(f"  birds eps={eps} m={m}: clusters {ref.max()+1}, border {int(((ref>=0)&~cs).sum())}, ambiguous border {amb}; noise differs in {ch} of 100 shuffles, the partition differs in {pd}; core set identical in all")

print("\n== Q2  k-means against density clusters")
k2 = KMeans(2, n_init=10, random_state=0).fit(X).labels_; d2 = DBSCAN(eps=.2, min_samples=5).fit(X).labels_
print("  birds: k-means ARI %.3f   DBSCAN ARI %.3f" % (ari(y, k2), ari(y, d2)))
Xo, yo = make_blobs(300, centers=[[0, 0], [3, 0], [1.5, 2.6]], cluster_std=1.0, random_state=0)
print("  three overlapping blobs (centres 3 apart, sd 1): k-means ARI %.3f" % ari(yo, KMeans(3, n_init=10, random_state=0).fit_predict(Xo)))
best = max(((ari(yo, DBSCAN(eps=e, min_samples=m).fit(Xo).labels_), e, m) for e in np.arange(0.1, 1.01, 0.05) for m in (3, 5, 10, 15)))
print("  best DBSCAN over eps 0.1..1.0 and m in 3,5,10,15: ARI %.3f at eps=%.2f m=%d" % best)
l = DBSCAN(eps=best[1], min_samples=best[2]).fit(Xo).labels_; print("    that run: clusters", l.max() + 1, " noise", int((l == -1).sum()))
for e in (0.3, 0.5, 1.0):
    l = DBSCAN(eps=e, min_samples=5).fit(Xo).labels_; print(f"  DBSCAN eps={e} m=5: clusters {l.max()+1}, noise {(l==-1).sum()}, ARI {ari(yo, l):.3f}")
print("  HDBSCAN(min_cluster_size=15): ARI %.3f" % ari(yo, HDBSCAN(min_cluster_size=15, copy=False).fit(Xo).labels_))

print("\n== Q3  min_cluster_size on the four groups of different density (min_samples 5)")
Xv, yv = varying()
for mc in (3, 5, 8, 20, 45, 100):
    h = HDBSCAN(min_samples=5, min_cluster_size=mc, copy=False).fit(Xv)
    t = h._single_linkage_tree_                     # private attribute, as in the chapter's figure script
    r = hdbscan_scratch(Xv, 5, mc, 'eom', Z=np.c_[t['left_node'], t['right_node'], t['value'], t['cluster_size']])
    assert np.array_equal(r['labels'], h.labels_)
    print(f"  m_c={mc:3d}: clusters {h.labels_.max()+1}, noise {(h.labels_==-1).sum():3d}, ARI {ari(yv, h.labels_):.3f}, condensed-tree clusters besides the root {len(r['S'])-1}")

print("\n== Q4  the reachability plot of the four groups (min_samples 8)")
o = OPTICS(min_samples=8, xi=0.1, min_cluster_size=20).fit(Xv); reach = o.reachability_
grp = {"tight group of 40": yv == 3, "tight group of 70 (left)": yv == 0, "tight group of 70 (right)": yv == 1, "diffuse group of 90": yv == 2}
for nm, g in grp.items():
    print(f"  {nm:26s} n={g.sum():2d}  median reachability {np.median(reach[g & np.isfinite(reach)]):.3f}  90th percentile {np.percentile(reach[g & np.isfinite(reach)], 90):.3f}")
print("  noise samples (uniform): median reachability %.3f" % np.median(reach[(yv == -1) & np.isfinite(reach)]))
for e in (0.15, 0.35, 0.8):
    l = cluster_optics_dbscan(reachability=o.reachability_, core_distances=o.core_distances_, ordering=o.ordering_, eps=e)
    per = {nm: (np.bincount(l[g] + 1, minlength=1).argmax() - 1, round(float((l[g] == -1).mean()), 2)) for nm, g in grp.items()}
    print(f"  cut at {e}: clusters {l.max()+1}, noise {(l==-1).sum()}, ARI {ari(yv, l):.3f}; cluster number of the four groups (tight 40, tight 70 left, tight 70 right, diffuse):", [int(v[0]) for v in per.values()], " noise share:", [v[1] for v in per.values()])
