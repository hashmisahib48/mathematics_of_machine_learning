# Numbers quoted in the prose of Chapter 37 (not a listing).
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering as AC

X = np.array([[3, 3], [1, 2], [0, 2], [7, 4], [11, 2],
              [5, 12], [8, 9]], float)
ss = lambda P: float(((P - P.mean(0)) ** 2).sum())
Z = linkage(X, "ward")
print("SS total of the 7 finches:", round(ss(X), 4))
print("sum of Delta over the 6 Ward merges:",
      round(float((Z[:, 2] ** 2 / 2).sum()), 4))
print("Ward heights:", Z[:, 2].round(3))
print("Ward Delta:", (Z[:, 2] ** 2 / 2).round(3))
print("gap ratio 11.18 / 4.472:", round(Z[4, 2] / Z[3, 2], 2))
for link in ("single", "complete", "average", "ward"):
    m = AC(n_clusters=3, linkage=link).fit(X)
    groups = {}
    for i, l in enumerate(m.labels_):
        groups.setdefault(l, []).append("ABCDEFG"[i])
    print(link, sorted("".join(g) for g in groups.values()))
# Delta of the six candidate merges after BC has been formed (Ex. 37.6-3)
P = {"BC": X[[1, 2]], "A": X[[0]], "D": X[[3]], "E": X[[4]]}
for a, b in (("BC", "A"), ("BC", "D"), ("BC", "E"), ("A", "D"),
             ("A", "E"), ("D", "E")):
    na, nb = len(P[a]), len(P[b])
    d2 = ((P[a].mean(0) - P[b].mean(0)) ** 2).sum()
    print(f"Delta({a},{b}) = {na * nb / (na + nb) * d2:.4f}")
# the five finches: SS of the pieces
print("SS(BC), SS(ABC):", ss(X[[1, 2]]), round(ss(X[[0, 1, 2]]), 4))
# mean pairwise Euclidean distances ABC vs DE
d = np.sqrt(((X[:3, None] - X[None, 3:5]) ** 2).sum(-1))
print("ABC-DE min, max, mean:", d.min().round(3), d.max().round(3),
      d.mean().round(3))
# swiss roll parameter range
print("swiss roll t range: [1.5 pi, 4.5 pi] =",
      round(1.5 * np.pi, 2), round(4.5 * np.pi, 2))

# five finches A-E: Ward heights, sum of Delta, SS total (Example 37.6-3)
from scipy.cluster.hierarchy import ward as _ward
X5 = X7[:5] if "X7" in dir() else np.array([[3,3],[1,2],[0,2],[7,4],[11,2]], float)
Z5 = _ward(X5)
print("A-E Ward heights:", Z5[:, 2].round(4), " Delta:", (Z5[:, 2]**2 / 2).round(4))
print("A-E SS total:", round(float(((X5 - X5.mean(0))**2).sum()), 4),
      " sum of Delta:", round(float((Z5[:, 2]**2 / 2).sum()), 4))
def _ss(P): return float(((P - P.mean(0))**2).sum())
print("A-E: SS(ABC), SS(DE):", round(_ss(X5[:3]), 4), round(_ss(X5[3:]), 4))
print("Delta(ABC,D) =", round(_ss(X5[[0,1,2,3]]) - _ss(X5[:3]), 4),
      " Delta(ABC,E) =", round(_ss(X5[[0,1,2,4]]) - _ss(X5[:3]), 4),
      " Delta(ABC,DE) =", round(_ss(X5) - _ss(X5[:3]) - _ss(X5[3:]), 4))

# average linkage by hand (Example 37.6-2)
_D = np.sqrt(((X5[:, None] - X5[None]) ** 2).sum(-1))
_dA = _D[0, 1:3].mean(); _dD = _D[3, 1:3].mean(); _dE = _D[4, 1:3].mean()
_aD = (1 * _D[0, 3] + 2 * _dD) / 3; _aE = (1 * _D[0, 4] + 2 * _dE) / 3
print("average: d(A,BC) %.4f d(D,BC) %.4f d(E,BC) %.4f" % (_dA, _dD, _dE))
print("average: d(ABC,D) %.4f d(ABC,E) %.4f  d(ABC,DE) %.4f"
      % (_aD, _aE, (_aD + _aE) / 2))
print("average: sum of the six distances %.3f" % (_D[:3, 3:].sum()),
      " mean %.4f" % _D[:3, 3:].mean())
