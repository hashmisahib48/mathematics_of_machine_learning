# Problem 18C.2: at most K - 1 discriminant directions
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
np.set_printoptions(precision=4, suppress=True)

rng = np.random.default_rng(0)
K, p, nk = 3, 1000, 20
means = rng.normal(size=(K, p)) * 0.2
X = np.vstack([means[k] + rng.normal(size=(nk, p)) for k in range(K)])
y = np.repeat(np.arange(K), nk)

pri = np.array([np.mean(y == k) for k in range(K)])
mu = np.array([X[y == k].mean(0) for k in range(K)])
xbar = pri @ mu                                   # overall mean
D = mu - xbar                                     # K rows mu_k - xbar
print("prior-weighted sum of the rows mu_k - xbar: norm", np.linalg.norm(pri @ D).round(12))
print("rank of the K x p matrix of rows mu_k - xbar:", np.linalg.matrix_rank(D))
Sb = (D.T * pri) @ D                              # p x p between-class scatter
print("S_b is", Sb.shape, " rank", np.linalg.matrix_rank(Sb), " eigenvalues, 4 largest:", np.sort(np.linalg.eigvalsh(Sb))[::-1][:4])
# the three class means span a plane: a third mean is an affine combination of the others
M = np.c_[mu[1] - mu[0], mu[2] - mu[0]]
print("rank of the two difference vectors mu_1 - mu_0, mu_2 - mu_0:", np.linalg.matrix_rank(M))

m = LDA().fit(X, y)
print("\nsolver svd: transform shape", m.transform(X).shape, " scalings_", m.scalings_.shape, " ratios", m.explained_variance_ratio_)
m = LDA(solver="eigen", shrinkage=0.5).fit(X, y)
print("solver eigen, shrinkage 0.5: scalings_", m.scalings_.shape, " transform", m.transform(X).shape)
print("explained_variance_ratio_ has", len(m.explained_variance_ratio_), "entries:", m.explained_variance_ratio_)
try: LDA(n_components=3).fit(X, y)
except ValueError as e: print("n_components=3:", type(e).__name__, "-", str(e)[:100])
y2 = (y > 0).astype(int)
print("two classes: transform shape", LDA().fit(X, y2).transform(X).shape)
