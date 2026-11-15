# KernelRidge on the three-point data set of Section 21.1.
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

X = np.array([[-1.0], [0.0], [1.0]])
y = np.array([1.0, 0.0, 1.0])

model = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(X, y)
print("dual_coef_  =", np.round(model.dual_coef_, 4))
print("X_fit_      =", model.X_fit_.ravel())
print("f(-1, 0, 1) =", np.round(model.predict(X), 4))
x_new = np.array([[0.5], [4.0]])
print("f(0.5), f(4) =", np.round(model.predict(x_new), 4))

# the same model from a precomputed Gram matrix
K = rbf_kernel(X, gamma=0.5)
pre = KernelRidge(alpha=0.1, kernel="precomputed").fit(K, y)
K_new = rbf_kernel(x_new, X, gamma=0.5)       # new rows against X
print("precomputed  =", np.round(pre.predict(K_new), 4))

# no intercept: a constant target is not reproduced far away
far = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
far.fit(X, np.full(3, 5.0))
print("y = 5 at the data:", np.round(far.predict(X), 3))
print("y = 5 at x = 10  :", np.round(far.predict([[10.0]]), 6))

# the default gamma is 1 / n_features
two = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
d = KernelRidge(kernel="rbf").fit(two, y)
e = KernelRidge(kernel="rbf", gamma=0.5).fit(two, y)
same = np.allclose(d.dual_coef_, e.dual_coef_)
print("gamma=None equals gamma=1/2:", same)
