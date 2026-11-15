# One epoch of SGD, hinge loss + L2 penalty (Example 17.7-2),
# repeated with SGDClassifier.partial_fit.
import numpy as np
from sklearn.linear_model import SGDClassifier

X = np.array([[4., 3], [1, 0], [3, 4], [0, 1]])
y = np.array([1, -1, 1, -1])
eta, alpha = 0.1, 1.0

def J(w, b):                           # Eq. 17.5-1, hinge, rho = 0
    z = y * (X @ w + b)
    return np.maximum(0, 1 - z).mean() + alpha / 2 * w @ w

w, b = np.zeros(2), 0.0
print(f"start      J = {J(w, b):.4f}")
for i in range(4):
    z = y[i] * (X[i] @ w + b)          # margin, Eq. 17.3-2
    w = (1 - eta * alpha) * w          # weight decay, Eq. 17.5-3
    if z <= 1:                         # hinge is active
        w = w + eta * y[i] * X[i]      # Eq. 17.5-3
        b = b + eta * y[i]
    print(f"i={i + 1}  z={z:+.4f}  w={w.round(4)}  b={b:+.4f}")
print(f"after one epoch  J = {J(w, b):.4f}")

clf = SGDClassifier(loss="hinge", penalty="l2", alpha=alpha,
                    learning_rate="constant", eta0=eta,
                    shuffle=False, random_state=0)
clf.partial_fit(X, y, classes=[-1, 1])    # one epoch, fixed order
print("scikit-learn:", clf.coef_[0].round(4),
      clf.intercept_.round(4), " t_ =", clf.t_)
