# Case study: GPS stops. Labels are hidden from the algorithms.
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import adjusted_rand_score as ari
from birds import gps

X, y = gps()                # 780 stops: 4 places and 80 strays
print("n =", len(X), " strays:", (y == -1).sum())

def report(name, lab):
    sizes = np.bincount(lab[lab >= 0]).tolist()
    print(f"{name:10s} k={lab.max() + 1} noise={(lab == -1).sum():3d}"
          f" ARI={ari(y, lab):.3f} {sizes}")

report("k-means", KMeans(4, n_init=10, random_state=0).fit(X).labels_)
for eps in (30, 60):
    report(f"DBSCAN {eps}", DBSCAN(eps=eps, min_samples=5).fit(X).labels_)
h = HDBSCAN(min_cluster_size=20, copy=False,
            store_centers="medoid").fit(X)
report("HDBSCAN", h.labels_)
bw = estimate_bandwidth(X)
report(f"MS h={bw:.0f}", MeanShift(bandwidth=bw).fit(X).labels_)
print("HDBSCAN strength, mean per cluster:",
      [round(float(h.probabilities_[h.labels_ == k].mean()), 2)
       for k in range(h.labels_.max() + 1)])
print("medoids:", h.medoids_.round().astype(int).tolist())
