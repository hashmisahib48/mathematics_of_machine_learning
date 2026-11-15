# Laplace approximation of Section 30.10, NumPy and scikit-learn.
import numpy as np
from scipy.special import expit, logit
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

X = np.array([[-3.0], [-2.0], [-1.0], [0.5], [1.0], [2.0], [3.0]])
y = np.array([0, 0, 0, 1, 0, 1, 1])
kern = ConstantKernel(4.0, "fixed") * RBF(1.5, "fixed")
K = kern(X)

f = np.zeros(len(y))                          # start at f = 0
for it in range(1, 30):                       # Newton, Eq. 30.10-4
    p = expit(f)
    W = p * (1 - p)                           # diagonal of W
    B = np.eye(len(y)) + np.sqrt(W)[:, None] * K * np.sqrt(W)
    Lc = np.linalg.cholesky(B)
    b = W * f + (y - p)
    a = b - np.sqrt(W) * np.linalg.solve(
        Lc.T, np.linalg.solve(Lc, np.sqrt(W) * (K @ b)))
    f_new = K @ a
    done = np.max(np.abs(f_new - f)) < 1e-10
    f = f_new
    if done:
        break
print("Newton steps:", it, "  mode:", np.round(f, 4))

gpc = GaussianProcessClassifier(kern).fit(X, y)
print("sklearn mode:", np.round(logit(gpc.base_estimator_.pi_), 4))

xs = np.array([[0.0], [1.5], [6.0]])
Ks = kern(X, xs)
mean = Ks.T @ (y - expit(f))                  # Eq. 30.10-6
V = np.linalg.solve(Lc, np.sqrt(W)[:, None] * Ks)
var = np.diag(kern(xs)) - (V**2).sum(0)       # Eq. 30.10-7
prob = [quad(lambda z: expit(z) * norm.pdf(z, m, np.sqrt(v)),
             -40, 40)[0] for m, v in zip(mean, var)]   # Eq. 30.10-8
print("latent mean:", np.round(mean, 4), " var:", np.round(var, 4))
print("by quadrature :", np.round(prob, 4))
print("predict_proba :", np.round(gpc.predict_proba(xs)[:, 1], 4))
