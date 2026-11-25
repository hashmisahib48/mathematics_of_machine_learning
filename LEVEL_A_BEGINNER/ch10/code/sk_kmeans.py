# The example of Section 10.4 with scikit-learn.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

X = np.array([[1, 1], [2, 1], [1, 2], [5, 4], [6, 5], [7, 4]])
m = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print("centers", m.cluster_centers_.round(4).tolist())
print("labels", m.labels_, " inertia", round(m.inertia_, 4))
print("distances of (2, 2) to the centers:",
      m.transform([[2, 2]]).round(4))
s = silhouette_samples(X, m.labels_)
mean = silhouette_score(X, m.labels_)
print("silhouettes", s.round(4), " mean", round(mean, 4))
