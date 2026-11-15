# Hand checks for Problems 17A.1 - 17A.4 and 17C.1 (every number quoted in the solutions)
import numpy as np
from sklearn.linear_model import Perceptron
np.set_printoptions(precision=4, suppress=True)
X = np.array([[4, 3], [1, 0], [3, 4], [0, 1]], float); y = np.array([1, -1, 1, -1.])

def run(order, eta=1.0, epochs=20, log=True):
    w = np.zeros(2); b = 0.0; ups = 0
    for e in range(1, epochs + 1):
        m = 0
        for i in order:
            z = y[i] * (X[i] @ w + b); act = z <= 0
            if log: print(f"epoch {e}  tile {i+1}  z = {z:+6.1f}  {'update' if act else 'keep  '}", end="")
            if act: w = w + eta * y[i] * X[i]; b = b + eta * y[i]; m += 1; ups += 1
            if log: print(f"   w = {w}  b = {b:+.1f}")
        if m == 0: break
    return w, b, ups

print("== 17A.1  order 3, 1, 4, 2"); w, b, ups = run([2, 0, 3, 1]); print("final w =", w, " b =", b, " updates =", ups)

print("\n== 17A.2  two lines, alpha = 0.1, rho = 0")
for name, w, b in (("(a)", np.array([2., 1.]), -3.0), ("(b)", np.array([.3, .3]), -1.2)):
    z = y * (X @ w + b); h = np.maximum(0, 1 - z); pen = 0.1 / 2 * (w @ w)
    print(name, "f =", X @ w + b, " z =", z, " hinge =", h, " mean hinge =", h.mean(), " |w|^2 =", round(w @ w, 4),
          " penalty =", round(pen, 4), " J =", round(h.mean() + pen, 4))

print("\n== 17A.3  two log-loss steps, eta = 0.5")
w = np.zeros(2); b = 0.0
for i in (0, 1):
    z = y[i] * (X[i] @ w + b); L = -1 / (1 + np.exp(z)); g = y[i] * L
    print(f"tile {i+1}: f = {X[i] @ w + b:.4f}  z = {z:.4f}  e^z = {np.exp(z):.4f}  L' = {L:.4f}  step = -eta*L'*y*x = {-0.5 * L * y[i] * X[i]}  step b = {-0.5 * L * y[i]:.4f}")
    w = w - 0.5 * g * X[i]; b = b - 0.5 * g; print("   w =", w, " b =", round(b, 4))

print("\n== 17A.4  optimal schedule, alpha = 1e-4")
a = 1e-4; eta1 = a ** -0.25; t0 = a ** -0.75 - 1
print("eta_1 = alpha^(-1/4) =", eta1, " alpha^(-3/4) =", a ** -0.75, " t0 =", t0, " 1/(alpha*eta1) - 1 =", 1 / (a * eta1) - 1)
for t in (1, 1000, 100000): print(f"eta_{t} = 1/(alpha (t + t0)) = {1 / (a * (t + t0)):.4f}   (alpha t + alpha^(1/4) - alpha = {a * t + a ** 0.25 - a:.6f})")
print("t at which eta falls to 1:", 1 / a - t0, " to 0.1:", 1 / (0.1 * a) - t0)

print("\n== 17C.1  Perceptron(shuffle=False) on the four tiles: the stopping rule of Section 17.5")
w = np.zeros(2); b = 0.0; best = np.inf; cnt = 0; tol = 1e-3; nic = 5
print("epoch  mistakes  sum of losses  best so far  counter")
for e in range(1, 30):
    m = 0; sl = 0.0
    for i in range(4):
        z = y[i] * (X[i] @ w + b); sl += max(0.0, -z)
        if z <= 0: w = w + y[i] * X[i]; b = b + y[i]; m += 1
    cnt = cnt + 1 if sl > best - tol * 4 else 0     # no gain of tol*n in the summed loss
    best = min(best, sl)
    print(f"{e:5d}  {m:8d}  {sl:13.1f}  {best:11.1f}  {cnt:7d}")
    if cnt >= nic: break
print("simulated n_iter =", e)
for k in (1, 2, 3, 5, 8):
    p = Perceptron(shuffle=False, n_iter_no_change=k).fit(X, y); print(f"n_iter_no_change = {k}: scikit-learn n_iter_ = {p.n_iter_}")
print("tol=None, max_iter=3:", Perceptron(shuffle=False, tol=None, max_iter=3).fit(X, y).n_iter_)

print("\n== extras quoted in the notes")
print("17A.2 widths 2/|w|: (a)", round(2 / np.sqrt(5), 3), " (b)", round(2 / np.sqrt(0.18), 3))
print("17A.2 J_a = 5 alpha/2, J_b = 0.1 + 0.18 alpha/2 are equal at alpha =", round(0.1 / (2.5 - 0.09), 4))
for al in (0.01, 0.0415, 0.1): print(f"   alpha = {al}: J_a = {2.5 * al:.4f}  J_b = {0.1 + 0.09 * al:.4f}")
print("17A.1 if z = 0 were not counted as a mistake:", end=" ")
w = np.zeros(2); b = 0.0; ups = 0
for e in range(10):
    m = 0
    for i in [2, 0, 3, 1]:
        if y[i] * (X[i] @ w + b) < 0: w = w + y[i] * X[i]; b += y[i]; m += 1
    ups += m
    if m == 0: break
print("w =", w, "b =", b, "updates", ups, " (the first sample is never updated, nothing moves)")
print("17A.1 final margins:", y * (X @ np.array([1., 2.]) + (-3.0)), " Example 17.7-1 final margins:", y * (X @ np.array([2., 1.]) + (-3.0)))
print("17A.3 sigma(-1.25) = 1/(1+e^-1.25) =", round(1 / (1 + np.exp(-1.25)), 4), " 1/(1+e^1.25)=", round(1 / (1 + np.exp(1.25)), 4))
print("17A.3 losses before/after:", end=" ")
def Jlog(w, b): return np.mean(np.log1p(np.exp(-y * (X @ w + b))))
print(round(Jlog(np.zeros(2), 0.), 4), round(Jlog(np.array([1., .75]), .25), 4), round(Jlog(np.array([.6114, .75]), -.1386), 4))

print("\n== 17A.4 read back from scikit-learn: one-hot samples, no penalty, update t moves only weight t by eta_t")
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
N = 100000; yy = np.where(np.arange(N) % 2 == 0, 1, -1)
m = SGDClassifier(loss="hinge", penalty=None, alpha=1e-4, learning_rate="optimal", fit_intercept=False, shuffle=False,
                  max_iter=1, tol=None, random_state=0).fit(sp.identity(N, format="csr"), yy)
for t in (1, 1000, 100000): print(f"scikit-learn eta_{t} = {abs(m.coef_[0][t - 1]):.4f}")

print("\n== 17A.3 read back from scikit-learn (one partial_fit on tiles 1 and 2) and 17B.1 norms")
m = SGDClassifier(loss="log_loss", penalty=None, learning_rate="constant", eta0=0.5, shuffle=False, random_state=0)
m.partial_fit(X[:2], (y[:2] + 1) // 2, classes=[0, 1]); print("scikit-learn: w =", m.coef_[0].round(4), " b =", m.intercept_.round(4))
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
Xc, yc = load_breast_cancer(return_X_y=True); Xc = StandardScaler().fit_transform(Xc); sc_ = 2.0 * yc - 1; a = 1e-3
w = np.zeros(30); b = 0.0; nrm = []
for t in range(1, 3 * 569 + 1):
    i = (t - 1) % 569; eta = 1 / (a ** 0.25 + a * (t - 1)); z = sc_[i] * (Xc[i] @ w + b); g = -sc_[i] * float(z <= 1)
    w = (1 - eta * a) * w - eta * g * Xc[i]; b -= eta * g; nrm.append(np.linalg.norm(w))
nrm = np.array(nrm); print("norm of the iterate at t = 1, 10, 100, 569, 1138, 1707:", nrm[[0, 9, 99, 568, 1137, 1706]].round(2), " max", nrm.max().round(2), "at t =", nrm.argmax() + 1, " mean of the norms", nrm.mean().round(2))
