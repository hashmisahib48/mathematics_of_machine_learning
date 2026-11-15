# Problem 36B.2: how often does one run reach the best inertia?
import numpy as np
from math import factorial
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(800, centers=8, cluster_std=0.9,
                  center_box=(-10, 10), random_state=4)
J = {}
for init in ("random", "k-means++"):
    J[init] = np.array([
        KMeans(8, init=init, n_init=1,
               random_state=s).fit(X).inertia_
        for s in range(50)])

best = min(J["random"].min(), J["k-means++"].min())  # of 100 runs
limit = 1.02 * best
print(f"best inertia {best:.2f}   2 per cent limit {limit:.2f}\n")
print("init       good  share    mean  median   worst")
for init, v in J.items():
    good = v < limit
    print(f"{init:9s} {good.sum():4d} {good.mean():6.2f}"
          f" {v.mean():7.1f} {np.median(v):7.1f} {v.max():7.1f}")

# 8 uniform seeds must hit 8 different blobs of 100 points each
print(f"\nP(8 random seeds in 8 blobs) = 8!/8^8 = "
      f"{factorial(8) / 8 ** 8:.4f}")
print(f"standard error of 0.68 from 50 runs: "
      f"{np.sqrt(0.68 * 0.32 / 50):.3f}")
