# HDBSCAN pipeline in NumPy/SciPy: mutual reachability, hierarchy,
# condensed tree, stability, excess of mass (Section 38.4).
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def hdbscan_scratch(X, min_samples=5, min_cluster_size=5,
                    method="eom", Z=None):
    X = np.asarray(X, float)
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    core = np.sort(D, axis=1)[:, min_samples - 1]       # Eq. 38.3-3
    MR = np.maximum(np.maximum(core[:, None], core[None]), D)  # 38.4-1
    np.fill_diagonal(MR, 0)
    if Z is None:
        Z = linkage(squareform(MR, checks=False), "single")
    size = lambda v: 1 if v < n else int(Z[v - n, 3])

    def leaves(v):
        out, st = [], [v]
        while st:
            u = st.pop()
            if u < n:
                out.append(u)
            else:
                st += [int(Z[u - n, 0]), int(Z[u - n, 1])]
        return out

    # condense the tree: rows (parent, child, lambda, size)
    rows, relabel, nxt, st = [], {2 * n - 2: n}, n + 1, [2 * n - 2]
    while st:
        v = st.pop(0)                                   # breadth first
        if v < n:
            continue
        a, b = int(Z[v - n, 0]), int(Z[v - n, 1])
        h = Z[v - n, 2]
        lam = 1 / h if h > 0 else np.inf                # Eq. 38.4-2
        p, sa, sb = relabel[v], size(a), size(b)
        if sa >= min_cluster_size and sb >= min_cluster_size:
            relabel[a], relabel[b], nxt = nxt, nxt + 1, nxt + 2
            rows += [(p, relabel[a], lam, sa), (p, relabel[b], lam, sb)]
            st += [a, b]
        else:
            for u, su in ((a, sa), (b, sb)):
                if su < min_cluster_size:               # falls out
                    rows += [(p, q, lam, 1) for q in leaves(u)]
                else:                                   # keeps going
                    relabel[u] = p
                    st.append(u)
    clusters = sorted({r[0] for r in rows})
    kids = {c: [r[1] for r in rows if r[0] == c and r[1] >= n]
            for c in clusters}
    birth = {n: 0.0}
    birth.update({r[1]: r[2] for r in rows if r[1] >= n})
    S = {c: sum((r[2] - birth[c]) * r[3] for r in rows if r[0] == c)
         for c in clusters}                             # Eq. 38.4-3

    keep, best = {}, dict(S)                            # Eq. 38.4-4
    for c in sorted(clusters, reverse=True):
        sub = sum(best[k] for k in kids[c])
        keep[c] = (not kids[c]) or (method == "eom" and sub <= S[c])
        if method == "eom" and kids[c] and sub > S[c]:
            best[c] = sub
    selected, todo = [], list(kids[n])                  # root excluded
    while todo:
        c = todo.pop(0)
        if keep[c]:
            selected.append(c)
        else:
            todo += kids[c]
    selected.sort()

    def subtree(c):
        out, st = [], [c]
        while st:
            u = st.pop()
            out.append(u)
            st += kids[u]
        return out

    labels = np.full(n, -1)
    prob = np.zeros(n)
    for j, c in enumerate(selected):
        sub = set(subtree(c))
        lam_max = max(r[2] for r in rows if r[0] == c)
        for r in rows:
            if r[0] in sub and r[1] < n:
                labels[r[1]] = j
                prob[r[1]] = (1.0 if not np.isfinite(r[2]) else
                              min(r[2], lam_max) / lam_max)
    return dict(labels=labels, prob=prob, rows=rows, S=S, kids=kids,
                selected=selected, Z=Z, core=core, MR=MR, n=n,
                birth=birth)
