import numpy as np
from sklearn.mixture import GaussianMixture
from gmm_scratch import MyGMM
from people import people_2d

X, _ = people_2d()
rng = np.random.default_rng(1)          # a rough common start
m0 = X[rng.choice(len(X), 3, replace=False)]
w0 = np.full(3, 1 / 3)
S0 = np.array([np.cov(X.T)] * 3)

mine = MyGMM(w0, m0, S0.copy()).fit(X)
ref = GaussianMixture(3, weights_init=w0, means_init=m0,
                      precisions_init=np.linalg.inv(S0)).fit(X)
for a, b in ((mine.pi_, ref.weights_), (mine.mu_, ref.means_),
             (mine.cov_, ref.covariances_),
             (mine.score_samples(X), ref.score_samples(X)),
             (mine.predict_proba(X), ref.predict_proba(X))):
    assert np.allclose(a, b, atol=1e-6)
assert mine.n_iter_ == ref.n_iter_
err = max(np.abs(mine.mu_ - ref.means_).max(),
          np.abs(mine.cov_ - ref.covariances_).max())
print("iterations    ", mine.n_iter_, ref.n_iter_)
print("mean log-lik  ", round(mine.lower_bound_, 6),
      round(ref.lower_bound_, 6))
print("largest difference in means, covariances:", f"{err:.1e}")
