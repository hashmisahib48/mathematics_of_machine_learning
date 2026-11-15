# Problem 38B.2: the scale of the wine columns.
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X, y, names = wine.data, wine.target, wine.feature_names
sd = X.std(axis=0)                 # sd of each column
share = sd**2 / (sd**2).sum()       # share of the total variance
j = sd.argmax()
print("largest sd %.2f (%s)" % (sd[j], names[j]))
print("smallest sd %.4f (%s)" % (sd.min(), names[sd.argmin()]))
print("largest / smallest: %.0f" % (sd.max() / sd.min()))
print("share of the largest column: %.4f" % share[j])
print("second largest share: %.5f" % np.sort(share)[-2])

Xs = StandardScaler().fit_transform(X)
v = Xs.var(0) / Xs.var(0).sum()
print("standardised: shares %.4f to %.4f, 1/13 = %.4f" %
      (v.min(), v.max(), 1 / 13))

# the distances of 13 raw columns are those of proline alone
r = np.corrcoef(pdist(X), pdist(X[:, [j]]))[0, 1]
print("correlation of pairwise distances, 13 columns vs proline: "
      "%.4f" % r)
for name, Z in (("13 columns", X), ("proline alone", X[:, [j]])):
    lab = DBSCAN(eps=40, min_samples=5).fit(Z).labels_
    print(f"DBSCAN eps=40 on {name}: {lab.max()+1} clusters, "
          f"{(lab == -1).sum()} noise, ARI {ari(y, lab):.3f}")
