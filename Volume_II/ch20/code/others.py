# Radius neighbours, nearest centroid and neighbour graphs.
import numpy as np
from sklearn.neighbors import (
    KNeighborsTransformer, NearestCentroid, NearestNeighbors,
    RadiusNeighborsClassifier)

X = np.array([[5, 3], [6, 2], [3, 4], [4, 1],
              [7, 5], [8, 8], [6, 9], [9, 8]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])     # 0 apricot, 1 plum
q = [[6.0, 5.0]]

rn = RadiusNeighborsClassifier(
    radius=3.2, outlier_label="most_frequent").fit(X, y)
inside = rn.radius_neighbors(q)[1][0] + 1
print("radius 3.2: points", np.sort(inside), " proba",
      rn.predict_proba(q)[0], " predict", rn.predict(q))
print("far query (0, 9): no neighbour, label",
      rn.predict([[0.0, 9.0]]))

nc = NearestCentroid().fit(X, y)
print("\ncentroids\n", nc.centroids_)
print("predict (7, 6):", nc.predict([[7.0, 6.0]]),
      " priors", nc.class_prior_)

graph = KNeighborsTransformer(n_neighbors=3, mode="distance")
G = graph.fit_transform(X)                 # sparse (8, 8) matrix
print("\nkneighbors graph", G.shape, "stored values", G.nnz)
print("row of point 8:", np.round(G[7].toarray()[0], 3))
nn = NearestNeighbors(n_neighbors=3).fit(X)
dist, idx = nn.kneighbors(X[[7]])
print("NearestNeighbors, point 8:", idx[0] + 1,
      np.round(dist[0], 3))
