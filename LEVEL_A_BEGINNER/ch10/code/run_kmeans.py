from kmeans import KMeans
X = [[1, 1], [2, 1], [1, 2], [5, 4], [6, 5], [7, 4]]
for n in (1, 2):
    m = KMeans(n_clusters=2, init=[[1, 1], [2, 1]], max_iter=n).fit(X)
    C = [[round(v, 4) for v in c] for c in m.cluster_centers_]
    print(f"after {n} iteration(s): centers {C}")
    print("   labels", m.labels_, " inertia", round(m.inertia_, 4))
m = KMeans(n_clusters=2, random_state=0).fit(X)
print("k-means++ inertia", round(m.inertia_, 4), "iter", m.n_iter_)
print("new points (2, 2) and (6, 3):", m.predict([[2, 2], [6, 3]]))
