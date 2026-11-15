import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from people import people_2d

X, _ = people_2d()
dp, dd = "dirichlet_process", "dirichlet_distribution"
runs = [(dp, None), (dd, 0.001), (dd, 10.0), (dd, 1000.0)]
print("prior    gamma0  active  sorted weights")
for kind, prior in runs:
    bgm = BayesianGaussianMixture(
        n_components=8, weight_concentration_prior_type=kind,
        weight_concentration_prior=prior, max_iter=500, tol=1e-6,
        random_state=0).fit(X)
    w = np.sort(bgm.weights_)[::-1]
    n = (w > 0.01).sum()
    print(f"{'DP' if kind == dp else 'Dir':<7s}",
          f"{bgm.weight_concentration_prior_:7.3f} {n:5d}   ",
          " ".join(f"{v:.2f}" for v in w))
