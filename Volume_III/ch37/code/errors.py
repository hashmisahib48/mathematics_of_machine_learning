# The error messages quoted in the text (not a listing).
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC

X = np.array([[3, 3], [1, 2], [0, 2], [7, 4], [11, 2]], float)
Z = np.vstack([X, [0, 0]])
cases = {
    "both n_clusters and threshold": lambda: AC(
        n_clusters=3, distance_threshold=4.0).fit(X),
    "affinity=": lambda: AC(affinity="l1").fit(X),
    "ward with manhattan": lambda: AC(metric="manhattan").fit(X),
    "cosine with a zero row": lambda: AC(
        metric="cosine", linkage="average").fit(Z),
    "full tree off with threshold": lambda: AC(
        n_clusters=None, distance_threshold=4.0,
        compute_full_tree=False).fit(X),
}
for name, f in cases.items():
    try:
        f()
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {e}")
