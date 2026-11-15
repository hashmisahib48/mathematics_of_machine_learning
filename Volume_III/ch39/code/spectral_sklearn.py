# Chapter 39 estimators on the toy graph and on the two rings.
import warnings
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari
from data import rings, toy_adjacency

sc = SpectralClustering(2, affinity="precomputed", random_state=0)
labels = sc.fit(toy_adjacency()).labels_
print("toy labels         ", labels.tolist())

X, truth = rings()                       # 300 points, labels hidden
km = KMeans(2, n_init=10, random_state=0).fit(X)
print("k-means ARI        ", round(ari(truth, km.labels_), 3))
rbf = SpectralClustering(2, affinity="rbf", gamma=30.0,
                         random_state=0).fit(X)
print("rbf gamma=30 ARI   ", round(ari(truth, rbf.labels_), 3))

with warnings.catch_warnings():     # "graph not fully connected"
    warnings.simplefilter("ignore", UserWarning)
    nn = SpectralClustering(2, affinity="nearest_neighbors",
                            n_neighbors=10, random_state=0).fit(X)
print("10 neighbours ARI  ", round(ari(truth, nn.labels_), 3))
n_comp = connected_components(nn.affinity_matrix_)[0]
print("graph components   ", n_comp)
for how in ("kmeans", "discretize", "cluster_qr"):
    m = SpectralClustering(2, gamma=30.0, assign_labels=how,
                           random_state=0).fit(X)
    print(how, "ARI", round(ari(truth, m.labels_), 3))
