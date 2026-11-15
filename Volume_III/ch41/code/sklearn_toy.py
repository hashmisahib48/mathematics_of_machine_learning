# scikit-learn's clustering scores on the eight customers.
import numpy as np
from sklearn import metrics

X = np.array([[1, 1], [2, 3], [3, 1], [4, 3],
              [7, 2], [8, 4], [3, 7], [5, 8]], float)
seg = np.array([0, 0, 0, 1, 1, 1, 2, 2])      # truth: labels_true
clu = np.array([0, 0, 0, 0, 1, 1, 2, 2])      # k-means, K = 3
for f in ["rand_score", "adjusted_rand_score", "mutual_info_score",
          "normalized_mutual_info_score", "adjusted_mutual_info_score",
          "homogeneity_score", "completeness_score",
          "v_measure_score", "fowlkes_mallows_score"]:
    print(f"{f:30s}{getattr(metrics, f)(seg, clu):.4f}")
for f in ["silhouette_score", "calinski_harabasz_score",
          "davies_bouldin_score"]:
    print(f"{f:30s}{getattr(metrics, f)(X, clu):.4f}")
