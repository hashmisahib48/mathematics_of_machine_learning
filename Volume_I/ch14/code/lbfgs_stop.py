# Why lbfgs stops at the start on an unscaled feature (Section 14.7).
import numpy as np
import scipy.optimize as so
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss.loss import HalfPoissonLoss

x = 1e5 * np.arange(5.0)[:, None]
y = np.array([1.0, 2.0, 2.0, 4.0, 6.0])
ll = LinearModelLoss(HalfPoissonLoss(), fit_intercept=True)
c0 = np.array([0.0, np.log(3.0)])
r = so.minimize(ll.loss_gradient, c0, method="L-BFGS-B", jac=True,
                options=dict(maxiter=100, maxls=50, gtol=1e-8,
                             ftol=64 * np.finfo(float).eps),
                args=(x, y, None, 0.0, 1))
print("message:", r.message, " status:", r.status, " nit:", r.nit)
g = ll.loss_gradient(c0, x, y, None, 0.0, 1)[1]
print("first gradient component:", g[0])
