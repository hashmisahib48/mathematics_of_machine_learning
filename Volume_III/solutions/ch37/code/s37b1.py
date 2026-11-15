# Problem 37B.1: four linkages on the irises, 3 clusters.
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import confusion_matrix

X, y = load_iris(return_X_y=True)   # no scaling: all in cm
print("linkage     ARI    sizes of the three clusters")
for link in ("single", "complete", "average", "ward"):
    lab = AC(n_clusters=3, linkage=link).fit_predict(X)
    sizes = np.sort(np.bincount(lab))[::-1]
    print(f"{link:9s} {ari(y, lab):6.3f}   {sizes}")

# species (rows) against clusters (columns)
for link in ("average", "ward", "single"):
    lab = AC(n_clusters=3, linkage=link).fit_predict(X)
    print(f"\n{link}: species (rows) x clusters (columns)")
    print(confusion_matrix(y, lab))
