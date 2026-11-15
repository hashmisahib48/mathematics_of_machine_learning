# Section 59.6: wine, exploration without labels.
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score as sil
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)      # y is used only to judge
raw = KMeans(3, n_init=10, random_state=0).fit_predict(X)
print(f"k-means on raw columns: ARI {ari(y, raw):.3f}")
Xs = StandardScaler().fit_transform(X)
pca = PCA(random_state=0).fit(Xs)
cum = pca.explained_variance_ratio_.cumsum()
print("variance kept by 2, 3, 5 components:",
      cum[[1, 2, 4]].round(3))
Z = PCA(2, random_state=0).fit_transform(Xs)

print(" K  silhouette    BIC")
for k in range(2, 7):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    bic = GaussianMixture(k, n_init=5, random_state=0).fit(Z).bic(Z)
    print(f"{k:2d}  {sil(Z, lab):9.3f} {bic:6.0f}")

fits = {
    "k-means": KMeans(3, n_init=10, random_state=0).fit(Z).labels_,
    "GMM": GaussianMixture(3, n_init=5,
                           random_state=0).fit_predict(Z),
    "HDBSCAN": HDBSCAN(min_cluster_size=10,
                       copy=True).fit_predict(Z),
}
for name, lab in fits.items():
    keep = lab >= 0
    print(f"{name:8s} clusters {lab.max() + 1}, "
          f"noise {(~keep).sum():3d}, "
          f"silhouette {sil(Z[keep], lab[keep]):.3f}, "
          f"ARI {ari(y, lab):.3f}")
