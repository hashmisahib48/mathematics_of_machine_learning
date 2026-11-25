# Clusters of iris flowers compared with the species.
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
Z = StandardScaler().fit_transform(X)
lab = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Z)
print(pd.crosstab(y, lab, rownames=["species"], colnames=["cluster"]))
print("adjusted Rand index", round(adjusted_rand_score(y, lab), 3))
