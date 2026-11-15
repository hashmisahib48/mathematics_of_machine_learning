# Problem 11B.3: a lasso path by coordinate descent, warm starts
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import lasso_path
from sklearn.model_selection import train_test_split

def soft(z, a):                              # Eq. 11.3-6
    return np.sign(z) * np.maximum(np.abs(z) - a, 0.0)

def cd_path(X, y, alphas, tol=1e-13):
    n, p = X.shape
    Xc, yc = X - X.mean(0), y - y.mean()
    v = (Xc ** 2).mean(0)                    # v_j, Eq. 11.3-3
    w, out = np.zeros(p), []                 # w carries over: warm
    for a in alphas:
        r = yc - Xc @ w                      # current residual
        for sweep in range(2000):
            big = 0.0
            for j in range(p):
                z = Xc[:, j] @ r / n + v[j] * w[j]  # Eq. 11.3-10
                new = soft(z, a) / v[j]
                r += Xc[:, j] * (w[j] - new)
                big = max(big, abs(new - w[j]))
                w[j] = new
            if big < tol: break
        out.append(w.copy())
    return np.array(out).T

d = load_diabetes()
Xtr, _, ytr, _ = train_test_split(d.data, d.target,
                                  test_size=0.25, random_state=0)
al = np.geomspace(2.4, 0.01, 30)             # 30 penalties
mine = cd_path(Xtr, ytr, al)
_, ref, _ = lasso_path(Xtr - Xtr.mean(0), ytr - ytr.mean(),
                       alphas=al, tol=1e-12, max_iter=100000)
print("max |difference| to lasso_path: %.1e" % abs(mine - ref).max())
k = (mine != 0).sum(0).tolist()
print("non-zero weights, penalties  1-15:", *k[:15])
print("non-zero weights, penalties 16-30:", *k[15:])
