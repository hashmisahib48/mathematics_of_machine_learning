# Lloyd, Elkan, mini-batch and bisecting k-means on one data set.
import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

X, _ = make_blobs(n_samples=300000, n_features=20, centers=15,
                  cluster_std=2.0, random_state=0)
K = 15
models = {
    "KMeans lloyd": KMeans(K, n_init=3, random_state=0),
    "KMeans elkan": KMeans(K, n_init=3, random_state=0,
                           algorithm="elkan"),
    "MiniBatch 1024": MiniBatchKMeans(K, n_init=3, random_state=0,
                                      batch_size=1024),
    "MiniBatch 4096": MiniBatchKMeans(K, n_init=3, random_state=0,
                                      batch_size=4096),
    "Bisecting": BisectingKMeans(K, n_init=3, random_state=0),
}
ref = None
print(f"{'model':16s}{'inertia':>11s}{'/ lloyd':>9s}"
      f"{'ARI vs lloyd':>14s}{'seconds':>9s}")
for name, m in models.items():
    t0 = time.perf_counter()
    m.fit(X)
    sec = time.perf_counter() - t0
    if ref is None:
        ref = m
    ari = adjusted_rand_score(ref.labels_, m.labels_)
    ratio = m.inertia_ / ref.inertia_
    print(f"{name:16s}{m.inertia_:11.1f}{ratio:9.4f}"
          f"{ari:14.4f}{sec:9.2f}")
mb = models["MiniBatch 1024"]
print("mini-batch steps:", mb.n_steps_, " passes:", mb.n_iter_)
print("lloyd iterations:", ref.n_iter_,
      " elkan iterations:", models["KMeans elkan"].n_iter_)
bis = models["Bisecting"]
same = (bis.transform(X).argmin(axis=1) == bis.labels_).mean()
print("bisecting: predict == labels_:",
      bool((bis.predict(X) == bis.labels_).all()),
      "; nearest centre == labels_:", f"{same:.4f}")
