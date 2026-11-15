# Case study: three kinds of wine, clustered without their labels.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

wine = load_wine()
X_raw, cultivar = wine.data, wine.target   # labels only to judge
X_std = StandardScaler().fit_transform(X_raw)
sd = X_raw.std(axis=0)
print(f"raw standard deviations: {sd.min():.2f} to {sd.max():.1f}")

for name, X in (("raw", X_raw), ("standardised", X_std)):
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    ari = adjusted_rand_score(cultivar, km.labels_)
    sil = silhouette_score(X, km.labels_)
    print(f"{name:13s} inertia {km.inertia_:10.2f}  ARI {ari:.3f}  "
          f"silhouette {sil:.3f}")
print(pd.crosstab(km.labels_, cultivar, rownames=["cluster"],
                  colnames=["cultivar"]))

print("K   inertia  silhouette   ARI")
for K in range(2, 9):
    km = KMeans(K, n_init=10, random_state=0).fit(X_std)
    sil = silhouette_score(X_std, km.labels_)
    ari = adjusted_rand_score(cultivar, km.labels_)
    print(f"{K}  {km.inertia_:8.2f}  {sil:9.3f}  {ari:6.3f}")
