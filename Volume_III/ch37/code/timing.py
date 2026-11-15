# Time of hierarchical clustering and BIRCH (machine dependent).
import json, time
import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.datasets import make_blobs

def best_of(fn, reps=3):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t)

sizes = [500, 1000, 2000, 4000, 8000]
out = {"n": sizes}
makers = {
    "ward": lambda X: AgglomerativeClustering(n_clusters=3),
    "average": lambda X: AgglomerativeClustering(
        n_clusters=3, linkage="average"),
    "single": lambda X: AgglomerativeClustering(
        n_clusters=3, linkage="single"),
    "birch": lambda X: Birch(threshold=0.8, n_clusters=3),
}
for name, mk in makers.items():
    out[name] = []
    for n in sizes:
        X, _ = make_blobs(n_samples=n, centers=3, random_state=0)
        out[name].append(round(best_of(lambda: mk(X).fit(X), 2), 4))
    print(f"{name:8s}", out[name])
allsizes = sizes + [10000, 100000]
out["bytes_condensed"] = {n: 8 * n * (n - 1) // 2 for n in allsizes}
json.dump(out, open("verify/timing.json", "w"), indent=1)
for n, b in out["bytes_condensed"].items():
    print(f"n = {n:6d}: condensed distance matrix {b / 1e9:8.3f} GB")
