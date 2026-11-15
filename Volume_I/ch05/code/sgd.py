# Stochastic gradient descent by hand, checked against SGDRegressor.
import numpy as np
from sklearn.linear_model import SGDRegressor

x = np.arange(1.0, 6.0)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[np.ones(5), x]                  # column of ones carries b


def epoch(eta, show=False):
    w = np.zeros(2)
    for t in range(5):                    # fixed order
        r = y[t] - X[t] @ w               # residual of sample t
        w = w + eta[t] * r * X[t]         # Eq. 5.4-3
        if show:
            print(f"t={t + 1}  r={r:+.4f}  w={w.round(4)}")
    return w


def library(**kw):
    m = SGDRegressor(penalty=None, shuffle=False, eta0=0.01, **kw)
    m.partial_fit(x[:, None], y)          # one epoch
    return np.r_[m.intercept_, m.coef_].round(4)


print("constant eta = 0.01")
w = epoch([0.01] * 5, show=True)
print("scikit-learn:", library(learning_rate="constant"))
print("invscaling, power_t = 0.25")
eta = 0.01 / np.arange(1, 6) ** 0.25      # Eq. 5.4-4
print("by hand:     ", epoch(eta).round(4))
print("scikit-learn:", library(learning_rate="invscaling",
                               power_t=0.25))
