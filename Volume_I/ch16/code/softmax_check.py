# Softmax regression: one sample, then iris against sklearn.
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

f = np.array([2.0, 1.0, 0.0])               # scores of one sample
P = softmax(f)                              # Eq. 16.4-1
Y = np.array([0.0, 1.0, 0.0])               # true class: 1
print("exp(f) =", np.round(np.exp(f), 4), "sum",
      round(np.exp(f).sum(), 4))
print("P      =", np.round(P, 4))
print(f"loss   = {-np.log(P[1]):.4f}")      # Eq. 16.4-2
print("P - Y  =", np.round(P - Y, 4), " sum",
      abs(round((P - Y).sum(), 9)))

X, y = load_iris(return_X_y=True)
n, p = X.shape
K, C = 3, 1.0
Y = np.eye(K)[y]                            # one-hot labels

def J(v):                                   # Eq. 16.5-1, K classes
    W, b = v[:K * p].reshape(K, p), v[K * p:]
    F = X @ W.T + b
    loss = (logsumexp(F, axis=1) - (F * Y).sum(1)).sum()
    return C * loss + 0.5 * (W ** 2).sum()

def grad(v):                                # Eq. 16.4-3
    W, b = v[:K * p].reshape(K, p), v[K * p:]
    R = softmax(X @ W.T + b, axis=1) - Y
    return np.r_[(C * R.T @ X + W).ravel(), C * R.sum(0)]


res = minimize(J, np.zeros(K * p + K), jac=grad,
               method="L-BFGS-B", options={"maxiter": 2000,
               "gtol": 1e-10, "ftol": 1e-15})
W, b = res.x[:K * p].reshape(K, p), res.x[K * p:]
ref = LogisticRegression(C=C, tol=1e-10, max_iter=1000).fit(X, y)
Pm = softmax(X @ W.T + b, axis=1)
err = abs(Pm - ref.predict_proba(X)).max()
print(f"max |P - predict_proba| = {err:.1e}")
print("column sums of coef_    =",
      np.round(ref.coef_.sum(0), 8) + 0.0)
