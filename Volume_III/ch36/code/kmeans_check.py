# The scratch class against scikit-learn on the scaled wine data.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from kmeans_scratch import MyKMeans, seed_pp

X = StandardScaler().fit_transform(load_wine().data)
for seed in range(5):
    mu0 = seed_pp(X, 3, np.random.default_rng(seed))   # same start
    mine = MyKMeans(3, init=mu0).fit(X)
    ref = KMeans(3, init=mu0, n_init=1, algorithm="lloyd").fit(X)
    assert np.array_equal(mine.labels_, ref.labels_)
    assert np.allclose(mine.cluster_centers_, ref.cluster_centers_)
    assert np.isclose(mine.inertia_, ref.inertia_)
    assert mine.n_iter_ == ref.n_iter_
    print(f"seed {seed}: inertia {mine.inertia_:.4f} = "
          f"{ref.inertia_:.4f}, iterations {mine.n_iter_}")
R = range(100)                  # seeding quality, single runs
J1 = [MyKMeans(3, random_state=s).fit(X).inertia_ for s in R]
J2 = [KMeans(3, n_init=1, random_state=s).fit(X).inertia_
      for s in R]
print(f"mean of 100 runs: scratch {np.mean(J1):.3f}, "
      f"scikit-learn {np.mean(J2):.3f}")
print("all assertions passed")
