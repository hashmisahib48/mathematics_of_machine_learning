# Which of the two eps = 1.4 answers (2.4037 or 2.4034) is nearer the minimum of Eq. 13.3-2?
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor
sys.path.insert(0, "../../../ch13/code")
from huber_scratch import MyHuber

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]); y = np.array([3.0, 4.0, 6.0, 7.0, 20.0]); eps = 1.4

def J(p):                                   # Eq. 13.3-2 with alpha = 0; p = (b, w, log sigma)
    b, w, ls = p; s = np.exp(ls); z = np.abs(y - b - w * x) / s
    h = np.where(z < eps, z ** 2, 2 * eps * z - eps ** 2)
    return np.sum(s + s * h)

me = MyHuber(epsilon=eps, alpha=0.0, n_iter=2000).fit(x[:, None], y)
sk = HuberRegressor(epsilon=eps, alpha=0, max_iter=2000, tol=1e-10).fit(x[:, None], y)
pm = [me.intercept_, me.coef_[0], np.log(me.scale_)]; ps = [sk.intercept_, sk.coef_[0], np.log(sk.scale_)]
print("J at MyHuber        %.10f  (w = %.5f)" % (J(pm), me.coef_[0]))
print("J at HuberRegressor %.10f  (w = %.5f)" % (J(ps), sk.coef_[0]))
best = minimize(J, pm, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000, maxfev=20000))
print("J at Nelder-Mead    %.10f  (w = %.5f)" % (best.fun, best.x[1]))
# slope of J along w at the two solutions (the valley is nearly flat in w)
for nm, p in (("MyHuber", pm), ("sklearn", ps)):
    d = 1e-4; print("%-8s dJ/dw = %.2e" % (nm, (J([p[0], p[1] + d, p[2]]) - J([p[0], p[1] - d, p[2]])) / (2 * d)))
