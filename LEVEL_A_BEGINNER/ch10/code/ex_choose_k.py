# Choosing k for the wine data: inertia and silhouette.
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_wine().data)
for k in range(2, 8):
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    s = silhouette_score(X, m.labels_)
    print(f"k = {k}: inertia {m.inertia_:7.1f}  silhouette {s:.3f}")
