# Case study: the digits data set, ten clusters, labels hidden.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, silhouette_score

X, digit = load_digits(return_X_y=True)   # digit: only to judge
print("images:", X.shape, " pixel range", X.min(), "to", X.max())
km = KMeans(10, n_init=10, random_state=0).fit(X)
print(f"inertia {km.inertia_:.1f}  iterations {km.n_iter_}")
print(f"ARI {adjusted_rand_score(digit, km.labels_):.3f}  "
      f"silhouette {silhouette_score(X, km.labels_):.3f}")
print("cluster  size  majority digit  share of that digit")
for k in range(10):
    m = km.labels_ == k
    counts = np.bincount(digit[m], minlength=10)
    print(f"{k:7d}  {m.sum():4d}  {counts.argmax():14d}"
          f"  {counts.max() / m.sum():18.2f}")
is8 = digit == 8
eights = [int(((km.labels_ == k) & is8).sum()) for k in range(10)]
print("the 174 eights, by cluster:", eights)
print("digits in cluster 1:", np.bincount(digit[km.labels_ == 1]))
