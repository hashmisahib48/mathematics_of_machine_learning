# Digits: choose the neighbourhood size of each method by T(10).
import json
import warnings
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_digits
from sklearn.manifold import (Isomap, LocallyLinearEmbedding,
                              SpectralEmbedding, trustworthiness)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
T10 = lambda Z: trustworthiness(Xtr, Z, n_neighbors=10)
res = {"Isomap": {}, "LLE": {}, "Laplacian": {}}
print("Isomap: n_neighbors, graph components, T(10)")
for m in (6, 8, 10, 12, 15, 20, 30):
    nc = connected_components(kneighbors_graph(Xtr, m))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Z = Isomap(n_neighbors=m).fit_transform(Xtr)
    res["Isomap"][m] = (int(nc), T10(Z))
    print(f"  {m:3d}  {nc}  {res['Isomap'][m][1]:.4f}")
print("LLE: n_neighbors, reg, T(10)")
for m in (6, 8, 10, 12, 16):
    for reg in (1e-3, 1e-1):
        Z = LocallyLinearEmbedding(n_neighbors=m, reg=reg,
                                   random_state=0).fit_transform(Xtr)
        res["LLE"][f"{m},{reg}"] = T10(Z)
        print(f"  {m:3d}  {reg:<6g}  {res['LLE'][f'{m},{reg}']:.4f}")
print("Laplacian eigenmaps: n_neighbors, T(10)")
for m in (8, 10, 15, 20, 30):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Z = SpectralEmbedding(n_neighbors=m, random_state=0
                              ).fit_transform(Xtr)
    res["Laplacian"][m] = T10(Z)
    print(f"  {m:3d}  {res['Laplacian'][m]:.4f}")
json.dump(res, open("verify/digits_scan.json", "w"), indent=1)
