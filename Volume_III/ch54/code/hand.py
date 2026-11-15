import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([[1, 20], [3, 50], [4, 40],
              [6, 70], [9, 140], [13, 100]], float)   # age, km
y = np.array([0, 0, 0, 1, 1, 1])
new = np.array([8.0, 80.0])

mu, sd = X.mean(0), X.std(0)              # Eq. 7.6-1, divide by n
Z = (X - mu) / sd
v = np.array([1, 1]) / np.sqrt(2)         # PC1, since rho > 0
s = Z @ v                                 # scores on PC1
cen = np.array([s[y == k].mean() for k in (0, 1)])
sn = ((new - mu) / sd) @ v
print("z1", Z[:, 0])
print("z2", Z[:, 1])
rho = (Z[:, 0] * Z[:, 1]).mean()
print("rho", round(rho, 4), " scores", s.round(3))
print("centroids", cen.round(4), " new score", round(sn, 4),
      "-> class", np.abs(sn - cen).argmin())

pipe = make_pipeline(StandardScaler(), PCA(1), NearestCentroid())
pipe.fit(X, y)
print("pipeline: class", pipe.predict([new])[0], " score",
      round(float(pipe[:2].transform([new])[0, 0]), 4))
print("centroids", pipe[-1].centroids_.round(4).ravel(), " PC1",
      pipe[1].components_.round(4).ravel(), " share",
      pipe[1].explained_variance_ratio_.round(4))
