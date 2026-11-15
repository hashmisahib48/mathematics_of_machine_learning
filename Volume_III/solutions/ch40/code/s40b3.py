# Problem 40B.3: a Dirichlet prior with small and large concentration
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import BayesianGaussianMixture

X, _ = make_blobs(n_samples=300, centers=2, random_state=3)
n, K = len(X), 6
for g0 in (0.01, 100.0):
    bgm = BayesianGaussianMixture(
        n_components=K, weight_concentration_prior=g0,
        weight_concentration_prior_type="dirichlet_distribution",
        max_iter=500, tol=1e-6, random_state=0).fit(X)
    w = np.sort(bgm.weights_)[::-1]
    Nk = np.sort(bgm.weight_concentration_ - g0)[::-1]  # soft counts
    print(f"gamma0 = {g0}: active (weight > 0.01):", (w > 0.01).sum())
    print("   weights", w.round(4).tolist())
    print("   soft counts N_k", Nk.round(2).tolist())
    # posterior mean weight (N_k + g0) / (n + K g0), Eq. 40.7-2
    post = (Nk + g0) / (n + K * g0)
    print("   (N_k+g0)/(n+K g0)", post.round(4).tolist())
