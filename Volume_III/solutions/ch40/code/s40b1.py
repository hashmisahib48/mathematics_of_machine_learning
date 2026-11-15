# Problem 40B.1: the BIC of the scratch class MyGMM (Listing 40.1)
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + '/../../../ch40/code')
import numpy as np
from sklearn.mixture import GaussianMixture
from gmm_scratch import MyGMM
from people import people_2d

X, _ = people_2d()                   # 500 people, p = 2
n, p = X.shape
K = 3
rng = np.random.default_rng(1)       # the start of Listing 40.2
m0 = X[rng.choice(n, K, replace=False)]
w0 = np.full(K, 1 / K)
S0 = np.array([np.cov(X.T)] * K)

mine = MyGMM(w0, m0, S0.copy()).fit(X)
ref = GaussianMixture(K, weights_init=w0, means_init=m0,
                      precisions_init=np.linalg.inv(S0)).fit(X)

c = K * p * (p + 1) // 2             # covariance numbers, full type
q = (K - 1) + K * p + c              # Eq. 40.6-1
ll = mine.score_samples(X).sum()     # maximised log-likelihood
bic = -2 * ll + q * np.log(n)        # Eq. 40.6-2
print("q =", K - 1, "+", K * p, "+", c, "=", q)
print(f"log-likelihood {ll:.3f}   mean {ll / n:.4f}")
print(f"BIC mine {bic:.4f}   BIC sklearn {ref.bic(X):.4f}")
print(f"AIC mine {-2 * ll + 2 * q:.4f}  AIC sklearn {ref.aic(X):.4f}")
print("iterations", mine.n_iter_, ref.n_iter_)

# mean log-likelihood after t EM steps (tol = 0 forces t steps)
for t in (1, 2, 5, 10, 19, 25):
    g = MyGMM(w0, m0, S0.copy(), tol=0, max_iter=t).fit(X)
    print(f"after {t:2d} steps: {g.score_samples(X).mean():.4f}")
