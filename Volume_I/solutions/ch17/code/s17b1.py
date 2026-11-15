# Problem 17B.1: hinge-loss SGD with the running mean of Eq. 17.5-8.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
s, a, n = 2.0 * y - 1, 1e-3, len(y)       # labels +-1, alpha

def eta(t):                               # "optimal", Eq. 17.5-7
    return 1 / (a ** 0.25 + a * (t - 1))

w, b, t = np.zeros(X.shape[1]), 0.0, 1
wbar, bbar = np.zeros_like(w), 0.0        # running means
for epoch in range(3):                    # fixed order, no shuffle
    for i in range(n):
        z = s[i] * (X[i] @ w + b)         # margin, Eq. 17.3-2
        g = -s[i] * float(z <= 1)         # dJ/df of the hinge loss
        w = (1 - eta(t) * a) * w - eta(t) * g * X[i]  # Eq. 17.5-3
        b = b - eta(t) * g
        wbar += (w - wbar) / t            # Eq. 17.5-8, a = 1
        bbar += (b - bbar) / t
        t += 1

ref = SGDClassifier(alpha=a, average=True, max_iter=3, tol=None,
                    shuffle=False).fit(X, y)
dw = np.abs(wbar - ref.coef_[0]).max()
db = abs(bbar - ref.intercept_[0])
print("updates:", t - 1)
print(f"largest |wbar - coef_|    : {dw:.1e}")
print(f"|bbar - intercept_|       : {db:.1e}")
print("norm of last iterate     :", round(np.linalg.norm(w), 4))
print("norm of the average      :", round(np.linalg.norm(wbar), 4))
acc = lambda u, c: round(np.mean(np.sign(X @ u + c) == s), 4)
print("training accuracy: last", acc(w, b),
      " average", acc(wbar, bbar))
