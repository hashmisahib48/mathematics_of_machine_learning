# New handwritten digits from a kernel density estimate.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
pca = PCA(20, random_state=0).fit(Xtr)        # 64 pixels -> 20
Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

grid = {"bandwidth": np.arange(2.0, 4.6, 0.5)}
gs = GridSearchCV(KernelDensity(), grid, cv=5).fit(Ztr)
kde = gs.best_estimator_                      # h from 5-fold CV
print("best bandwidth", kde.bandwidth)
print("mean ln f: train", round(kde.score(Ztr) / len(Ztr), 2),
      " test", round(kde.score(Zte) / len(Zte), 2))

gm = GaussianMixture(10, reg_covar=1e-3, n_init=3, random_state=0)
gm.fit(Ztr)                                   # Chapter 40
print("mixture of 10:  test", round(gm.score(Zte), 2))

Znew = kde.sample(500, random_state=0)        # Eq. 47.6-1
img = pca.inverse_transform(Znew)
print("pixel range of 500 samples", img.min().round(1),
      img.max().round(1))
s = kde.score_samples(Zte)                    # ln f(z) per digit
low = np.argsort(s)[:10]                      # least likely digits
print("lowest scores", s[low[0]].round(1), s[low[-1]].round(1))
print("their digits ", yte[low])
