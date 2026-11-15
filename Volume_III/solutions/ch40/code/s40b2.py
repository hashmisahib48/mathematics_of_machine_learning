# Problem 40B.2: k-means and a mixture on stretched, tilted groups
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=500, centers=3, random_state=170)
A = np.array([[0.60834549, -0.63667341],
              [-0.40887718, 0.85253229]])
X = X @ A                              # stretch and tilt every group

km = KMeans(3, n_init=10, random_state=0).fit(X)
gm = GaussianMixture(3, n_init=5, random_state=0).fit(X)

def table(lab):                # rows: true group, columns: found
    return np.array([[np.sum((y == i) & (lab == j)) for j in range(3)]
                     for i in range(3)])

fits = (("k-means", km.labels_), ("mixture", gm.predict(X)))
for name, lab in fits:
    T = table(lab)
    r, c = linear_sum_assignment(-T)   # match found to true groups
    print(f"{name}: ARI {adjusted_rand_score(y, lab):.3f},"
          f" wrong {T.sum() - T[r, c].sum()} of {len(y)}")
    print(T)

# shape of each fitted group: ratio of the axes of its ellipse
for k in np.argsort(gm.means_[:, 0]):
    ev = np.linalg.eigvalsh(gm.covariances_[k])
    print(f"component {k}: weight {gm.weights_[k]:.3f}"
          f" axis ratio {np.sqrt(ev[1] / ev[0]):.2f}")
sv = np.linalg.svd(A, compute_uv=False)
print("singular values of A:", sv.round(3),
      "ratio", (sv[0] / sv[1]).round(2))
