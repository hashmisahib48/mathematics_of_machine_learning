# Hand checks for Problems 38A.1 - 38A.3 (numbers quoted in the solutions).
import numpy as np
from sklearn.cluster import DBSCAN, cluster_optics_dbscan

print("== 38A.1  tags on a road, eps = 1.5, m = 3")
x = np.array([0, 1, 2, 3.5, 5, 6, 9.]); D = np.abs(x[:, None] - x[None]); eps, m = 1.5, 3
for i in range(7):
    nb = [j + 1 for j in range(7) if D[i, j] <= eps]
    print(f"  tag {i+1} at {x[i]:3.1f}: neighbourhood {nb}  count {len(nb)}  {'core' if len(nb) >= m else ''}")
cnt = (D <= eps).sum(1); core = cnt >= m
print("  distances equal to eps:", [(i + 1, j + 1) for i in range(7) for j in range(i + 1, 7) if D[i, j] == eps])
border = [i + 1 for i in range(7) if not core[i] and (core & (D[i] <= eps)).any()]; noise = [i + 1 for i in range(7) if not core[i] and i + 1 not in border]
print("  core", [int(i) + 1 for i in np.flatnonzero(core)], " border", border, " noise", noise)
cc = [(int(i) + 1, int(j) + 1, float(D[i, j])) for i in np.flatnonzero(core) for j in np.flatnonzero(core) if i < j and D[i, j] <= eps]; print("  core-core links:", cc)
lab = DBSCAN(eps=eps, min_samples=m).fit(x[:, None]).labels_; print("  DBSCAN labels", lab.tolist(), " core samples (1-based)", (DBSCAN(eps=eps, min_samples=m).fit(x[:, None]).core_sample_indices_ + 1).tolist())
print("  sum of the counts", cnt.sum(), "= 7 + 2 * pairs within eps:", 7 + 2 * sum(D[i, j] <= eps for i in range(7) for j in range(i + 1, 7)))

print("\n== 38A.2  mutual reachability, m = 3")
x2 = np.array([0, 1, 1.5, 4.]); n2 = "abcd"; D2 = np.abs(x2[:, None] - x2[None]); c3 = np.sort(D2, 1)[:, 2]
for i in range(4): print(f"  {n2[i]}: sorted distances {np.sort(D2[i]).tolist()}  c_3 = {c3[i]}")
MR = np.maximum(D2, np.maximum(c3[:, None], c3[None]))
pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
for i, j in pairs: print(f"  d_mr({n2[i]}{n2[j]}) = max(d {D2[i,j]}, c {c3[i]}, c {c3[j]}) = {MR[i,j]}")
def mst(W):                                  # Kruskal on the six pairs
    par = list(range(4)); f = lambda a: a if par[a] == a else f(par[a]); out = []
    for i, j in sorted(pairs, key=lambda p: (W[p], p)):
        if f(i) != f(j): par[f(i)] = f(j); out.append((n2[i] + n2[j], float(W[i, j])))
    return out
t1, t2 = mst(MR), mst(D2); print("  MST of d_mr:", t1, " total", sum(w for _, w in t1)); print("  MST of d   :", t2, " total", sum(w for _, w in t2))
print("  lambda = 1/height:", [round(1 / w, 4) for _, w in t1])

print("\n== 38A.3  reachability plot")
reach = np.array([np.inf, .3, .4, 1.2, .3, .35, .9, .4]); core4 = np.array([.3, .3, .4, .5, .3, .3, .45, .4])
for e in (0.5, 1.0, 1.3):
    lab = -np.ones(8, int); k = -1
    for p in range(8):                       # the rule of Section 38.5 along the ordering
        if reach[p] > e and core4[p] <= e: k += 1; lab[p] = k
        elif reach[p] <= e: lab[p] = k
    ref = cluster_optics_dbscan(reachability=reach, core_distances=core4, ordering=np.arange(8), eps=e)
    print(f"  eps={e}: labels {lab.tolist()}  library {ref.tolist()}  agree {np.array_equal(lab, ref)}")

print("\n== slips")
cs = (D < eps).sum(1); print("  38A.1 with a strict test (distance < 1.5): counts", cs.tolist(), "-> core tags", [int(i) + 1 for i in np.flatnonzero(cs >= m)])
cn = (D - np.eye(7) * 0 <= eps).sum(1) - 1; print("  38A.1 forgetting the tag itself: counts", cn.tolist(), "-> core tags", [int(i) + 1 for i in np.flatnonzero(cn >= m)])
c3wrong = np.sort(D2, 1)[:, 3]; print("  38A.2 third distance counted without the self-zero (slip):", c3wrong.tolist())
