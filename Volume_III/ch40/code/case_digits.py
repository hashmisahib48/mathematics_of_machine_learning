import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
pca = PCA(20, random_state=0).fit(Xtr)        # 64 pixels -> 20
Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
print("variance kept", pca.explained_variance_ratio_.sum().round(3))

gm = GaussianMixture(10, reg_covar=1e-3, n_init=3, random_state=0)
gm.fit(Ztr)                                   # labels are not used
print("converged", gm.converged_, " iterations", gm.n_iter_)
print("mean log-lik: train", round(gm.score(Ztr), 2),
      " test", round(gm.score(Zte), 2))
ari = adjusted_rand_score(yte, gm.predict(Zte))
print("adjusted Rand on the test digits", round(ari, 3))

Znew, _ = gm.sample(500)                      # invent new digits
img = pca.inverse_transform(Znew)
print("pixel range of 500 samples", img.min().round(1),
      img.max().round(1))
s = gm.score_samples(Zte)                     # ln p(z) per digit
low = np.argsort(s)[:10]                      # least likely digits
print("lowest scores", s[low[0]].round(1), s[low[-1]].round(1))
print("their digits ", yte[low])
