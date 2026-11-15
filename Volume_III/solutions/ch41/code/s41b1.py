# Problem 41B.1: k-means on the irises, external scores and
# the silhouette for K = 2 ... 6
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn import metrics as m

X, y = load_iris(return_X_y=True)
lab = KMeans(3, n_init=10, random_state=0).fit_predict(X)

print("contingency table (rows: species, columns: clusters)")
print(m.cluster.contingency_matrix(y, lab))          # n_ij
for name, f in [("ARI", m.adjusted_rand_score),
                ("AMI", m.adjusted_mutual_info_score),
                ("homogeneity", m.homogeneity_score),
                ("completeness", m.completeness_score),
                ("Fowlkes-Mallows", m.fowlkes_mallows_score)]:
    print(f"{name:16s}{f(y, lab):7.3f}")

print("\n K  silhouette   ARI   cluster sizes")
for K in range(2, 7):
    lk = KMeans(K, n_init=10, random_state=0).fit_predict(X)
    print(f"{K:2d}{m.silhouette_score(X, lk):10.3f}"
          f"{m.adjusted_rand_score(y, lk):9.3f}   "
          f"{sorted(np.bincount(lk).tolist())}")
print("\nsilhouette of the species themselves:",
      round(m.silhouette_score(X, y), 3))
