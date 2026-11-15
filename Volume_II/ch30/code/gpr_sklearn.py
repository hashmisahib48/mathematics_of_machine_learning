# GaussianProcessRegressor on the six settings of Section 30.1.
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

X = np.array([[1.0], [2.0], [3.0], [5.0], [6.0], [8.0]])
y = np.array([0.2, 1.1, 1.5, 0.4, -0.6, 0.3])

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(
    kernel, alpha=0.01, n_restarts_optimizer=5, random_state=0)
gp.fit(X, y)

print("kernel_ :", gp.kernel_)
print("theta   :", np.round(gp.kernel_.theta, 4), "(logs)")
print("LML     :", round(gp.log_marginal_likelihood_value_, 4))
Xs = np.array([[0.0], [4.0], [4.5], [7.0], [9.0], [12.0]])
mean, std = gp.predict(Xs, return_std=True)
for x_, m_, s_ in zip(Xs[:, 0], mean, std):
    print(f"x = {x_:5.1f}   mean = {m_:6.3f}   std = {s_:.3f}")
print("alpha_  :", np.round(gp.alpha_, 3))
print("L_ shape:", gp.L_.shape, "  R2 on training data:",
      round(gp.score(X, y), 4))
draws = gp.sample_y(Xs, n_samples=3, random_state=1)
print("sample_y shape:", draws.shape)
