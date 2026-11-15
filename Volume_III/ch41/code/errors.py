# Three traps of the clustering scores (the first one is a warning).
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import adjusted_rand_score, completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

X = np.array([[1, 1], [2, 3], [3, 1], [4, 3],
              [7, 2], [8, 4], [3, 7], [5, 8]], float)
seg = np.array([0, 0, 0, 1, 1, 1, 2, 2])
clu = KMeans(3, n_init=10, random_state=0).fit_predict(X)
print("k-means ids:", clu)
print("accuracy", accuracy_score(seg, clu),
      " ARI", round(adjusted_rand_score(seg, clu), 4))
print("homogeneity(truth, clusters)  ",
      round(homogeneity_score(seg, clu), 4))
print("homogeneity(clusters, truth)  ",
      round(homogeneity_score(clu, seg), 4), "= completeness",
      round(completeness_score(seg, clu), 4))
for f in (silhouette_score, calinski_harabasz_score,
          davies_bouldin_score):
    for labels in (np.zeros(8, int), np.arange(8)):
        try:
            f(X, labels)
        except ValueError as err:
            print(f.__name__, "ValueError:", err)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    from sklearn.metrics.cluster import entropy
    entropy(seg)
print(w[0].category.__name__ + ":", w[0].message)
