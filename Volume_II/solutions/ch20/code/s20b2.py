# Problem 20B.2: RadiusNeighborsClassifier on the moons of 20.11
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier

X, y = make_moons(300, noise=0.3, random_state=0)
Xa, Xb, ya, yb = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
share = np.bincount(ya).max() / len(ya)
print("most frequent training class:", np.bincount(ya).argmax(),
      " share", round(share, 3))
print("  r  accuracy  no neighbour  mean neighbours  per unit area")
for r in (0.1, 0.2, 0.4, 0.8):
    m = RadiusNeighborsClassifier(radius=r,
                                  outlier_label="most_frequent")
    m.fit(Xa, ya)
    nb = m.radius_neighbors(Xb, return_distance=False)
    size = np.array([len(v) for v in nb])
    dens = size.mean() / (np.pi * r ** 2)
    empty = (size == 0).sum()
    print(f"{r:4.1f}  {m.score(Xb, yb):7.4f}  {empty:12d}"
          f"  {size.mean():15.1f}  {dens:13.1f}")
    if r == 0.1:     # the empty queries get the most frequent class
        e = size == 0
        hit = (m.predict(Xb[e]) == yb[e]).mean()
        print("      empty queries: true classes",
              np.bincount(yb[e], minlength=2),
              " accuracy on them", round(float(hit), 3))
