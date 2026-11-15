import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 2], [2, 1], [3, 3], [8, 9], [9, 8], [10, 10]])
y = np.array([3, 4, 6, 15, 16, 19])          # a target
scaler = StandardScaler()                    # a transformer
Z = scaler.fit_transform(X)                  # learn, then apply
print("mean_ :", scaler.mean_.round(3),
      " scale_:", scaler.scale_.round(3))
reg = LinearRegression().fit(X, y)           # supervised: needs y
print("coef_ :", reg.coef_.round(4),
      " R^2:", round(reg.score(X, y), 4))
km = KMeans(n_clusters=2, n_init=10, random_state=0)
labels = km.fit_predict(X)                   # unsupervised: no y
print("labels:", labels, " inertia_:", round(km.inertia_, 3))
print("transform shape:", km.transform(X).shape)  # distances
