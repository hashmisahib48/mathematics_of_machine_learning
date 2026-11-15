# The class of Listing 30.2 against scikit-learn (no optimisation).
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from gp_scratch import MyGP

rng = np.random.default_rng(0)
X = rng.uniform(0, 8, (25, 1))
y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=25)
Xs = np.linspace(-1, 9, 60)[:, None]

mine = MyGP(sf2=0.8, ell=1.3, noise=0.04).fit(X, y)
ref = GaussianProcessRegressor(
    ConstantKernel(0.8, "fixed") * RBF(1.3, "fixed"),
    alpha=0.04, optimizer=None).fit(X, y)

m1, s1 = mine.predict(Xs, return_std=True)
m2, s2 = ref.predict(Xs, return_std=True)
_, C1 = mine.predict(Xs, return_cov=True)
_, C2 = ref.predict(Xs, return_cov=True)
lml2 = ref.log_marginal_likelihood_value_
pairs = [("mean", m1, m2), ("std", s1, s2), ("cov", C1, C2),
         ("lml", mine.lml_, lml2)]
for name, a, b in pairs:
    assert np.allclose(a, b, atol=1e-8), name
    print(f"{name}: max |difference| = {np.max(np.abs(a - b)):.1e}")
print("log marginal likelihood:", round(mine.lml_, 4))
