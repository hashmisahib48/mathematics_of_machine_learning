# Wine: internal and external scores; the cultivars only judge.
from sklearn import metrics as m
from sklearn.cluster import AgglomerativeClustering as Agg, KMeans
from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
km = lambda Z: KMeans(3, n_init=10, random_state=0).fit_predict(Z)
agg = lambda link: Agg(n_clusters=3, linkage=link).fit_predict(Xs)
gm = GaussianMixture(3, n_init=5, tol=1e-5, max_iter=500,
                     random_state=0).fit_predict(Xs)


def row(name, Z, lab):
    print(f"{name:12s}{m.silhouette_score(Z, lab):7.3f}"
          f"{m.calinski_harabasz_score(Z, lab):8.1f}"
          f"{m.davies_bouldin_score(Z, lab):7.3f} |"
          f"{m.adjusted_rand_score(y, lab):7.3f}"
          f"{m.adjusted_mutual_info_score(y, lab):7.3f}")


print("            silh.     CH     DB |    ARI    AMI")
for name, lab in [("truth", y), ("k-means", km(Xs)),
                  ("GMM full", gm), ("Ward", agg("ward")),
                  ("complete", agg("complete")),
                  ("single", agg("single"))]:
    row(name, Xs, lab)
row("k-means raw", X, km(X))
