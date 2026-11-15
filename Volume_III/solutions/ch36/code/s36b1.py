# Problem 36B.1: inertia of standardised wine data, K = 1 ... 6
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_wine().data)
n, p = X.shape              # each column: mean 0, variance 1
print("n =", n, " p =", p, " n*p =", n * p)

J = {}
for K in range(1, 7):
    km = KMeans(K, n_init=10, random_state=0).fit(X)
    J[K] = km.inertia_

print("\n K   inertia    drop   drop/J(K-1)")
print(f"{1:2d} {J[1]:9.1f}")
for K in range(2, 7):
    d = J[K - 1] - J[K]           # what the K-th centre gains
    share = d / J[K - 1]
    print(f"{K:2d} {J[K]:9.1f} {d:7.1f} {share:9.3f}")

# K = 1: the centre is the column mean 0, so J = sum of all x_ij^2
print("\nsum of all squared entries:", round((X ** 2).sum(), 4))
print("J(1) equals n*p (SST):", np.isclose(J[1], n * p))
