# Problem 37C.1: labelling a new sample after Ward clustering (the data of the book's figure).
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

X, _ = make_blobs(n_samples=90, centers=[[0, 0], [4, 0], [2, 3.4]], cluster_std=0.6, random_state=2)
m = AC(n_clusters=3).fit(X); lab = m.labels_; new = np.array([[1.9, 1.0]])
print("has predict:", hasattr(m, "predict"), "  attributes:", [a for a in ("labels_", "children_", "distances_", "n_leaves_") if hasattr(m, a)])
print("children_ shape", m.children_.shape, " (merges of the 90 training samples only)")
sz = np.bincount(lab); cen = np.array([X[lab == k].mean(0) for k in range(3)])
d2 = ((cen - new) ** 2).sum(1); cost = sz / (sz + 1) * d2
print("cluster sizes", sz.tolist(), " centroids", cen.round(2).tolist())
print("squared distances to the centroids", d2.round(3).tolist(), "-> nearest centroid:", int(d2.argmin()))
print("Ward costs n/(n+1) d^2           ", cost.round(3).tolist(), "-> cheapest:", int(cost.argmin()))
for k in (1, 5, 15):
    print(f"{k:2d}-nearest neighbours votes:", np.bincount(KNeighborsClassifier(k).fit(X, lab).predict(new), minlength=3).tolist(), " label", int(KNeighborsClassifier(k).fit(X, lab).predict(new)[0]))
print("logistic regression on the labels:", int(LogisticRegression().fit(X, lab).predict(new)[0]))
Xn = np.r_[X, new]; l2 = AC(n_clusters=3).fit(Xn).labels_
back = int(np.bincount(lab[l2[:-1] == l2[-1]]).argmax())      # old cluster that the new sample's refit cluster mostly consists of
print("refit with the new sample: it joins the cluster that was number", back, " old labels of the 90 samples changed:", len(set(zip(lab, l2[:-1]))) != 3)
# how often does refitting change the old samples? random new points in the box
rng = np.random.default_rng(0); ch = 0
for _ in range(200):
    p = rng.uniform([-1, -1], [5, 4.4]); l3 = AC(n_clusters=3).fit(np.r_[X, [p]]).labels_[:-1]; ch += len(set(zip(lab, l3))) != 3
print("200 random new points: refit moves some old sample in", ch, "cases")
