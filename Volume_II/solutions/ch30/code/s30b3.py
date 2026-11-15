# Problem 30B.3: normalize_y with readings near 100.
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

X = np.array([1., 2, 3, 5, 6, 8])[:, None]
y = np.array([0.2, 1.1, 1.5, 0.4, -0.6, 0.3]) + 100.0
for flag in (False, True):
    gp = GPR(C(1.0) * RBF(1.5), alpha=0.01, normalize_y=flag,
             n_restarts_optimizer=3, random_state=0).fit(X, y)
    m, s = gp.predict([[30.0], [4.0]], return_std=True)
    print(f"normalize_y={flag!s:5}  kernel_ = {gp.kernel_}")
    print(f"  LML {gp.log_marginal_likelihood_value_:.3f}")
    print(f"  x=30: mean {m[0]:.3f} std {s[0]:.3f}")
    print(f"  x=4:  mean {m[1]:.3f} std {s[1]:.3f}")
n, sd = len(y), y.std()
print(f"mean of y {y.mean():.3f}, std of y {sd:.3f}")
# normalize_y reports the likelihood of (y - mean)/std; y itself
# differs by the Jacobian n ln(std):
print("LML in the units of y:",
      round(gp.log_marginal_likelihood_value_ - n * np.log(sd), 3))
# direct check: fixed kernel scaled to the units of y
k = C(gp.kernel_.k1.constant_value * sd**2, "fixed") * \
    RBF(gp.kernel_.k2.length_scale, "fixed")
g = GPR(k, alpha=0.01 * sd**2, optimizer=None).fit(X, y - y.mean())
print("direct fit of y - mean: LML",
      round(g.log_marginal_likelihood_value_, 3))
