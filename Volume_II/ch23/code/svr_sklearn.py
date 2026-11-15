# SVR on the delivery data and on a noisy sine curve.
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVR

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
lin = SVR(kernel="linear", C=1.0, epsilon=1.0).fit(X, y)
print("support_      ", lin.support_)
print("n_support_    ", lin.n_support_)
print("dual_coef_    ", lin.dual_coef_.round(4))
print("coef_, b      ", lin.coef_.round(4), lin.intercept_.round(4))
print("f(6), score   ", lin.predict([[6.0]]).round(4),
      round(lin.score(X, y), 4))

rng = np.random.default_rng(4)          # the sine of Chapter 21
xs = np.sort(rng.uniform(0, 6, 28))
ys = np.sin(xs) + 0.25 * rng.normal(size=28)
Xs = xs[:, None]
m = SVR(C=10.0, epsilon=0.2).fit(Xs, ys)    # gamma="scale"
g = 1.0 / (1 * Xs.var())             # Eq. 22.5-4 (Chapter 22)
print("gamma by hand ", round(g, 4))
print("support vectors", m.n_support_[0], "of", len(ys),
      "- dual_coef_ shape", m.dual_coef_.shape)
print("sum of dual_coef_ is 0:", abs(m.dual_coef_.sum()) < 1e-9,
      "- largest |dual|", np.abs(m.dual_coef_).max().round(4))
K = rbf_kernel(Xs, m.support_vectors_, gamma=g)
by_hand = K @ m.dual_coef_[0] + m.intercept_[0]    # Eq. 23.3-6
err = np.abs(by_hand - m.predict(Xs)).max()
print("max |hand - predict| < 1e-10:", err < 1e-10)
print("n_iter_, fit_status_", m.n_iter_, m.fit_status_)
print("has coef_?    ", hasattr(m, "coef_"))
