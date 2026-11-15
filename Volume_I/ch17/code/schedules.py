# Learning rates eta_t of scikit-learn, read off one update at a
# time (§17.5). With five one-hot samples, update t moves
# only w_t, by exactly eta_t, so coef_ is the list eta_1 ... eta_5.
import numpy as np
from sklearn.linear_model import SGDClassifier

X, y = np.eye(5), np.ones(5)            # hinge is active at w = 0
alpha, eta0, p = 0.01, 0.1, 0.5
t = np.arange(1, 6)

def eta_from_sklearn(**kw):
    clf = SGDClassifier(penalty=None, shuffle=False,
                        fit_intercept=False, alpha=alpha,
                        eta0=eta0, power_t=p, **kw)
    return clf.partial_fit(X, y, classes=[0, 1]).coef_[0]

typw = np.sqrt(1 / np.sqrt(alpha))      # Eq. 17.5-6
eta1 = typw                             # first optimal rate
formula = {
    "constant":   np.full(5, eta0),                 # Eq. 17.5-4
    "invscaling": eta0 / t ** p,                    # Eq. 17.5-5
    "optimal":    1 / (1 / eta1 + alpha * (t - 1)),  # Eq. 17.5-7
}
for name, f in formula.items():
    s = eta_from_sklearn(learning_rate=name)
    print(f"{name:11s}", s.round(4), np.allclose(s, f))
t0 = 1 / (eta1 * alpha) - 1             # t0 of the documentation
print("eta_1 = alpha**(-1/4) =", round(alpha ** -0.25, 4),
      "  t0 =", round(t0, 2))
