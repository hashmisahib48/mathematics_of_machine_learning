# Agglo against AgglomerativeClustering: tree, heights, labels.
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from agglo_scratch import Agglo

def pairs(c):                       # a merge is an unordered pair
    return sorted(map(tuple, np.sort(c, axis=1)))

finch = np.array([[3, 3], [1, 2], [0, 2], [7, 4], [11, 2],
                  [5, 12], [8, 9]], float)
wine = StandardScaler().fit_transform(load_wine().data)
for name, X in (("finches", finch), ("wine", wine)):
    for link in ("single", "complete", "average", "ward"):
        a = Agglo(link, 3).fit(X)
        b = AgglomerativeClustering(n_clusters=3, linkage=link,
                                    compute_distances=True).fit(X)
        assert pairs(a.children_) == pairs(b.children_)
        assert np.allclose(a.distances_, b.distances_)
        assert len(set(zip(a.labels_, b.labels_))) == 3
        print(f"{name:8s}{link:9s}ok", np.bincount(a.labels_))
