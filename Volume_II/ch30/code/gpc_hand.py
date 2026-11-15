# Example 30.10-1: two labelled settings, Laplace mode by hand.
import numpy as np
from scipy.special import expit
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

X = np.array([[-1.0], [1.0]]); y = np.array([0, 1])
kern = ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")
K = kern(X); rho = K[0, 1]
print("rho = k(-1, 1) =", round(rho, 5), " 1 - rho =", round(1 - rho, 5))
f = np.zeros(2)
for it in range(1, 8):
    p = expit(f); W = p * (1 - p)
    f = np.linalg.solve(np.linalg.inv(K) + np.diag(W), W * f + y - p)
    print(f"step {it}: f = {np.round(f, 5)}  pi = {np.round(expit(f), 5)}")
c = f[1]; print("c =", round(c, 5), " 1 - sigmoid(c) =", round(1 - expit(c), 5))
# first step in closed form
a1 = 0.5 / (1 + 0.25 * (1 - rho)); print("a1 =", round(a1, 5), " f1 =", round((1 - rho) * a1, 5))
for xs in (0.0, 2.0):
    ks = kern(X, np.array([[xs]]))[:, 0]
    mean = ks @ (y - expit(f)); W = expit(f) * (1 - expit(f))
    Wr = np.sqrt(W); Bm = np.eye(2) + Wr[:, None] * K * Wr
    v = np.linalg.solve(np.linalg.cholesky(Bm), Wr * ks)
    var = 1.0 - v @ v
    pq = quad(lambda z: expit(z) * norm.pdf(z, mean, np.sqrt(var)), -40, 40)[0]
    ph = expit(mean / np.sqrt(1 + np.pi * var / 8))
    print(f"x*={xs}: k*={np.round(ks, 5)} mean={mean:.5f} var={var:.5f} "
          f"p(quad)={pq:.4f} p(MacKay)={ph:.4f} sigma(mean)={expit(mean):.4f}")
g = GaussianProcessClassifier(kern).fit(X, y)
print("sklearn predict_proba:", np.round(g.predict_proba([[0.0], [2.0]])[:, 1], 4))
print("LML sklearn:", round(g.log_marginal_likelihood_value_, 5))
