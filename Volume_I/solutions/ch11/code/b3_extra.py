# Extra numbers for Problem 11B.3: where do the sweeps go along the path?
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def sweeps_per_alpha(X, y, alphas, warm, tol):
    n, p = X.shape; Xc, yc = X - X.mean(0), y - y.mean(); v = (Xc ** 2).mean(0)
    w = np.zeros(p); out = []
    for a in alphas:
        if not warm: w = np.zeros(p)
        r = yc - Xc @ w
        for s in range(1, 2001):
            big = 0.0
            for j in range(p):
                z = Xc[:, j] @ r / n + v[j] * w[j]; new = np.sign(z) * max(abs(z) - a, 0) / v[j]
                r += Xc[:, j] * (w[j] - new); big = max(big, abs(new - w[j])); w[j] = new
            if big < tol: break
        out.append(s)
    return out
d = load_diabetes(); Xtr, _, ytr, _ = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
al = np.geomspace(2.4, 0.01, 30)
for tol in (1e-6, 1e-13):
    wm, cd = sweeps_per_alpha(Xtr, ytr, al, True, tol), sweeps_per_alpha(Xtr, ytr, al, False, tol)
    print("tol", tol)
    if tol > 1e-10: print("alpha :", np.round(al, 3).tolist()); print("warm  :", wm); print("cold  :", cd)
    print("total sweeps: warm %d, cold %d" % (sum(wm), sum(cd)))
    print("first 15 penalties: warm %d, cold %d ; last 15: warm %d, cold %d" % (sum(wm[:15]), sum(cd[:15]), sum(wm[15:]), sum(cd[15:])))
