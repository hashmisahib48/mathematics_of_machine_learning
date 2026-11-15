# PCA of the five fish with scikit-learn.
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[27, 4], [29, 6], [30, 6], [31, 8], [33, 6]], float)
new = np.array([[32.0, 5.0]])                  # a sixth fish

pca = PCA(n_components=2).fit(X)
print("mean_               ", pca.mean_)
print("components_ =\n", pca.components_.round(4))
print("singular_values_    ", pca.singular_values_.round(4))
print("explained_variance_ ", pca.explained_variance_.round(4))
print("ratio_              ",
      pca.explained_variance_ratio_.round(4))
print("scores, first col.  ", pca.transform(X)[:, 0].round(4))

one = PCA(n_components=1).fit(X)               # keep one component
z = one.transform(new)
print("new fish: score", z.round(4), " back",
      one.inverse_transform(z).round(4))
print("noise_variance_     ", one.noise_variance_)
print("score_samples(new)  ", one.score_samples(new).round(4))

white = PCA(n_components=2, whiten=True).fit(X)
print("whitened variances  ",
      white.transform(X).var(axis=0, ddof=1).round(4))
print("n_components=0.9 ->", PCA(0.9).fit(X).n_components_,
      " 0.8 ->", PCA(0.8).fit(X).n_components_)
