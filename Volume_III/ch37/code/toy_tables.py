# Distance matrices after each merge (single, complete) and the Ward
# update, for the five finches of Example 37.6-1 to 37.6-3.
import numpy as np
from scipy.spatial.distance import pdist, squareform
from agglo_scratch import LW

X = np.array([[3, 3], [1, 2], [0, 2], [7, 4], [11, 2]], float)
names = ["A", "B", "C", "D", "E"]
for link in ("single", "complete", "ward"):
    D = squareform(pdist(X)) ** (2 if link == "ward" else 1)
    np.fill_diagonal(D, np.inf)
    nm, size = list(names), [1.0] * 5
    print(link)
    steps = 2 if link == "ward" else 4
    for s in range(steps):
        a, b = divmod(D.argmin(), len(nm))
        h = D[a, b] ** (0.5 if link == "ward" else 1)
        extra = f"  Delta {D[a, b] / 2:.4f}" if link == "ward" else ""
        print("  merge", nm[a] + nm[b], "at", round(h, 4) , extra)
        keep = [k for k in range(len(nm)) if k not in (a, b)]
        ai, aj, be, ga = LW[link](size[a], size[b], np.array(
            [size[k] for k in keep]))
        new = (ai * D[a, keep] + aj * D[b, keep] + be * D[a, b]
               + ga * abs(D[a, keep] - D[b, keep]))
        D2 = np.full((len(keep) + 1, len(keep) + 1), np.inf)
        D2[1:, 1:] = D[np.ix_(keep, keep)]
        D2[0, 1:] = D2[1:, 0] = new
        nm = [nm[a] + nm[b]] + [nm[k] for k in keep]
        size = [size[a] + size[b]] + [size[k] for k in keep]
        D = D2
        print("   ", nm)
        print(np.where(np.isinf(D), 0, D).round(3))
