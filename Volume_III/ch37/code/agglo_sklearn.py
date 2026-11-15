# AgglomerativeClustering on the seven finches.
import numpy as np
from scipy.cluster.hierarchy import ward
from sklearn.cluster import AgglomerativeClustering

X = np.array([[3, 3], [1, 2], [0, 2], [7, 4], [11, 2],
              [5, 12], [8, 9]], float)
model = AgglomerativeClustering(
    n_clusters=3, linkage="ward", compute_distances=True).fit(X)
print("labels_    ", model.labels_)
print("children_  ", model.children_.tolist())
print("distances_ ", model.distances_.round(3))

same = np.allclose(model.distances_, ward(X)[:, 2])
print("heights equal scipy ward:", same)

def n_at(t):
    m = AgglomerativeClustering(n_clusters=None, linkage="ward",
                                distance_threshold=t).fit(X)
    return int(m.n_clusters_)

ts = (2.0, 4.0, 4.5, 12.0, 15.0)
print("thresholds", ts, "->", [n_at(t) for t in ts])
