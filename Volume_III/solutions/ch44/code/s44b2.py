# Problem 44B.2: connectivity and quality of Isomap on the S-curve
import warnings
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap, trustworthiness
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

X, t = make_s_curve(800, random_state=0)
print(f"{'m':>3s}{'components':>11s}{'T(10)':>8s}{'m-th NN':>9s}"
      f"{'across':>8s}")
for m in (3, 5, 10, 30):
    graph = kneighbors_graph(X, m)                 # graph of Isomap
    n_comp = connected_components(graph)[0]
    with warnings.catch_warnings():                # m = 3 warns and
        warnings.simplefilter("ignore")            # joins the pieces
        Z = Isomap(n_neighbors=m).fit_transform(X)
    T = trustworthiness(X, Z, n_neighbors=10)
    d, idx = NearestNeighbors(n_neighbors=m + 1).fit(X).kneighbors(X)
    dt = np.abs(t[idx[:, 1:]] - t[:, None])        # along the curve
    across = (dt > np.pi).mean()                   # long edges
    print(f"{m:3d}{n_comp:11d}{T:8.4f}{d[:, -1].mean():9.3f}"
          f"{across:8.4f}")

# the nearest approach of distant parts of the curve
D = np.linalg.norm(X[:, None] - X[None], axis=2)
far = np.abs(t[:, None] - t[None]) > np.pi       # far along curve
print("smallest distance between such points:", D[far].min().round(3))
