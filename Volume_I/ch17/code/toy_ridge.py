# RidgeClassifier on the four tiles (§17.6).
import numpy as np
from sklearn.linear_model import RidgeClassifier

X = np.array([[4., 3], [1, 0], [3, 4], [0, 1]])
y = np.array([1, -1, 1, -1])
alpha = 2.0

xbar, tbar = X.mean(axis=0), y.mean()   # centring, Eq. 17.6-2
Xc, tc = X - xbar, y - tbar
A = Xc.T @ Xc + alpha * np.eye(2)       # X'X + alpha I
w = np.linalg.solve(A, Xc.T @ tc)       # Eq. 17.6-2
b = tbar - xbar @ w
print("X'X =", (Xc.T @ Xc).tolist(), " X't =", (Xc.T @ tc).tolist())
print("by hand: w =", w.round(4), " b =", round(b, 4))
s = X @ w + b                           # scores, Eq. 17.3-1
print("scores  :", s.round(4), "-> classes", np.sign(s).astype(int))

clf = RidgeClassifier(alpha=alpha).fit(X, y)
print("sklearn : w =", clf.coef_.round(4), " b =",
      clf.intercept_.round(4), " shape", clf.coef_.shape)
print("decision_function:", clf.decision_function(X).round(4))
