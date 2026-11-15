# The ten birds with the four estimators of this chapter.
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, MeanShift

X = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [1, 2],
              [6, 1], [7, 1], [7, 2], [8, 2], [4, 5]], float)
fmt = lambda a: " ".join(f"{v:g}" for v in np.round(a, 3))

db = DBSCAN(eps=1.5, min_samples=4).fit(X)
print("DBSCAN labels ", fmt(db.labels_))
print("core samples  ", fmt(db.core_sample_indices_ + 1))
print("components_   ", db.components_.shape)

hd = HDBSCAN(min_cluster_size=4, min_samples=3, copy=False).fit(X)
print("HDBSCAN labels", fmt(hd.labels_))
print("probabilities ", fmt(hd.probabilities_))

op = OPTICS(min_samples=4, cluster_method="dbscan", eps=1.5).fit(X)
print("OPTICS order  ", fmt(op.ordering_ + 1))
print("reachability  ", fmt(op.reachability_))
print("core distances", fmt(op.core_distances_))
print("OPTICS labels ", fmt(op.labels_))

ms = MeanShift(bandwidth=1.5).fit(X)
print("centres       ", ms.cluster_centers_.round(1).tolist())
print("MeanShift     ", fmt(ms.labels_))
print("predict       ", fmt(ms.predict([[2, 2], [5, 5]])))
