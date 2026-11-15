# Birch: partial_fit equals fit; the global step is Ward.
import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.datasets import make_blobs

centres = np.array([[0, 0], [5, 0], [2.5, 4.5]])
X, _ = make_blobs(n_samples=600, centers=centres, cluster_std=0.9,
                  random_state=0)
full = Birch(threshold=1.0, n_clusters=None).fit(X)
part = Birch(threshold=1.0, n_clusters=None)
for chunk in np.array_split(X, 6):
    part.partial_fit(chunk)
a, b = full.subcluster_centers_, part.subcluster_centers_
print("subclusters: fit", len(a), " six partial_fit calls", len(b),
      " same centres:", np.allclose(a, b))

birch = Birch(threshold=1.0, n_clusters=3).fit(X)
ward = AgglomerativeClustering(n_clusters=3)
ward.fit(birch.subcluster_centers_)
print("global step is Ward on the centres:",
      np.array_equal(ward.labels_, birch.subcluster_labels_))
print("predict(X) equals labels_:",
      np.array_equal(birch.predict(X), birch.labels_))
for t in (0.3, 0.8, 1.5):
    fit = Birch(threshold=t, n_clusters=None).fit(X)
    n = len(fit.subcluster_centers_)
    print(f"threshold {t}: {n} subclusters")
