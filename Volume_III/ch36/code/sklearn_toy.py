# KMeans on the eight customers: fit, read, predict.
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[3, 7], [1, 3], [1, 4], [3, 8],
              [4, 0], [2, 1], [5, 8], [1, 8]], float)
start = np.array([[0, 5], [5, 5]], float)       # as in Example 36.7-1
km = KMeans(n_clusters=2, init=start, n_init=1, algorithm="lloyd",
            max_iter=300, tol=1e-4).fit(X)
print("centres:", km.cluster_centers_.tolist())
print("labels :", km.labels_, " inertia:", km.inertia_,
      " iterations:", km.n_iter_)

new = np.array([[2.0, 6.0], [4.0, 2.0]])        # two new customers
print("predict:", km.predict(new))
print("distances to the depots:\n", km.transform(new).round(4))
print("score(new) =", km.score(new), " (minus the inertia of new)")
print("feature names:", km.get_feature_names_out())

default = KMeans(n_clusters=2, random_state=0).fit(X)   # k-means++
print("default n_init:", default.n_init, " inertia:", default.inertia_)
