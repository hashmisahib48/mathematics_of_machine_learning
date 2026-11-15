# Hierarchical clustering of the wine data; labels only judge.
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
print("ARI against the cultivars, K = 3")
print(f"{'':20s}{'raw':>7s}{'scaled':>8s}   sizes (scaled)")
variants = [("single", "euclidean"), ("complete", "euclidean"),
            ("average", "euclidean"), ("ward", "euclidean"),
            ("average", "cosine")]
for link, metric in variants:
    row = []
    for Z in (X, Xs):
        model = AgglomerativeClustering(
            n_clusters=3, linkage=link, metric=metric).fit(Z)
        row.append(ari(y, model.labels_))
    name = link if metric == "euclidean" else f"{link} ({metric})"
    sizes = np.sort(np.bincount(model.labels_))[::-1]
    print(f"{name:20s}{row[0]:7.3f}{row[1]:8.3f}   {sizes}")
km = [KMeans(3, n_init=10, random_state=0).fit_predict(Z)
      for Z in (X, Xs)]
print(f"{'k-means':20s}{ari(y, km[0]):7.3f}{ari(y, km[1]):8.3f}")
