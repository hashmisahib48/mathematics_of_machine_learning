# Problem 36B.3: three ways of scaling the wine data
import numpy as np
from itertools import permutations
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X, y = load_wine(return_X_y=True)
sd = X.std(0)
print("column std, smallest and largest:", sd.min().round(2),
      sd.max().round(1))

scalers = {"none": None, "StandardScaler": StandardScaler(),
           "MinMaxScaler": MinMaxScaler()}
print("\nscaling             ARI    inertia  widest column")
labels = {}
for name, sc in scalers.items():
    Z = X if sc is None else sc.fit_transform(X)
    km = KMeans(3, n_init=10, random_state=0).fit(Z)
    labels[name] = km.labels_
    ari = adjusted_rand_score(y, km.labels_)
    share = Z.var(0).max() / Z.var(0).sum()   # share of variance
    print(f"{name:14s} {ari:6.3f} {km.inertia_:10.2f} {share:8.3f}")

def errors(lab):                  # wines in the wrong cluster
    return min(int((np.array(q)[lab] != y).sum())
               for q in permutations(range(3)))

a, b = labels["StandardScaler"], labels["MinMaxScaler"]
print("\nwines misplaced: standard", errors(a),
      " min-max", errors(b))
print("ARI between the two clusterings:",
      round(adjusted_rand_score(a, b), 3))
